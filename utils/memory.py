"""
MemoryStore — хранилище диалогов с опциональным векторным поиском.

Два режима:
- **full**: SQLite + FAISS + SentenceTransformer (для дома, есть torch)
- **lite**: только SQLite (для облака на 1 GB RAM, без ML-зависимостей)

Режим определяется автоматически: если `faiss`/`sentence_transformers` не
установлены или `vector_search=False`, переходим в lite. `search()` тогда
работает по SQL LIKE — менее точно, но без 400 MB RAM на эмбеддер.
"""

import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Опциональные ML-зависимости — могут отсутствовать в lite-режиме
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    logger.info("faiss недоступен — векторный поиск отключён")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class MemoryStore:
    """Хранилище диалогов. Vector search опционален."""

    DEFAULT_EMBED_DIM = 384

    def __init__(
        self,
        path: str,
        embed_model: str = "all-MiniLM-L6-v2",
        max_records: int = 50000,
        vector_search: bool = True,
    ):
        self.path = path
        self.max_records = max_records

        # Решаем — включать ли vector search
        self.vector_search = (
            vector_search
            and FAISS_AVAILABLE
            and NUMPY_AVAILABLE
            and SENTENCE_TRANSFORMERS_AVAILABLE
        )

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_db()

        self.embedder = None
        self.embedding_dim = self.DEFAULT_EMBED_DIM
        self.index = None
        self.idx_file = path.replace(".sqlite", ".faiss")

        if self.vector_search:
            self._init_embedder(embed_model)
            self._init_index()
        else:
            logger.info("MemoryStore: lite-mode (SQLite only, без vector search)")

    # ---------- инициализация ----------

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                bot TEXT NOT NULL,
                important INTEGER DEFAULT 0,
                timestamp REAL NOT NULL,
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS vector_map (
                memory_id INTEGER PRIMARY KEY,
                vector_id INTEGER NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memory(id)
            );

            CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory(timestamp);
            CREATE INDEX IF NOT EXISTS idx_memory_important ON memory(important);
            """
        )
        self.conn.commit()

    def _init_embedder(self, embed_model: str) -> None:
        try:
            self.embedder = SentenceTransformer(embed_model)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            self.embedder = None
            self.vector_search = False

    def _init_index(self) -> None:
        if not self.vector_search:
            return
        if os.path.exists(self.idx_file):
            try:
                self.index = faiss.read_index(self.idx_file)
                logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)
                return
            except Exception as e:
                logger.error("Failed to load FAISS index: %s", e)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info("Created new FAISS index")

    # ---------- публичный API ----------

    def add(self, user_text: str, bot_response: str, important: bool = False) -> int:
        timestamp = time.time()
        embedding = self._encode(user_text) if self.vector_search else None

        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memory (user, bot, important, timestamp, embedding) VALUES (?, ?, ?, ?, ?)",
            (user_text, bot_response, int(important), timestamp, embedding),
        )
        memory_id = cursor.lastrowid

        if embedding and self.vector_search:
            vector_id = self.index.ntotal
            self.index.add(np.frombuffer(embedding, dtype=np.float32).reshape(1, -1))
            cursor.execute(
                "INSERT INTO vector_map (memory_id, vector_id) VALUES (?, ?)",
                (memory_id, vector_id),
            )
            faiss.write_index(self.index, self.idx_file)

        self.conn.commit()
        self._check_limits()
        return memory_id

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск похожих диалогов. Vector search если доступен, иначе SQL LIKE."""
        if self.vector_search and self.index is not None and self.index.ntotal > 0:
            return self._search_vector(query, top_k)
        return self._search_like(query, top_k)

    def _search_vector(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            query_vec = self.embedder.encode(query, convert_to_numpy=True)
            faiss.normalize_L2(query_vec.reshape(1, -1))
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vec.reshape(1, -1), k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                cursor = self.conn.execute(
                    """
                    SELECT m.id, m.user, m.bot, m.timestamp
                    FROM memory m JOIN vector_map vm ON m.id = vm.memory_id
                    WHERE vm.vector_id = ?
                    """,
                    (int(idx),),
                )
                row = cursor.fetchone()
                if row:
                    results.append({
                        "id": row[0],
                        "user": row[1],
                        "bot": row[2],
                        "timestamp": row[3],
                        "score": float(dist),
                    })
            return results
        except Exception as e:
            logger.error("Vector search error: %s", e)
            return []

    def _search_like(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback-поиск по SQL LIKE — для lite-режима."""
        if not query.strip():
            return []
        like = f"%{query.strip().lower()}%"
        cursor = self.conn.execute(
            """
            SELECT id, user, bot, timestamp
            FROM memory
            WHERE LOWER(user) LIKE ? OR LOWER(bot) LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (like, like, top_k),
        )
        return [
            {"id": r[0], "user": r[1], "bot": r[2], "timestamp": r[3], "score": 0.5}
            for r in cursor
        ]

    def fetch(self, indices: List[int]) -> List[str]:
        if not indices:
            return []
        placeholders = ",".join("?" * len(indices))
        cursor = self.conn.execute(
            f"""
            SELECT m.user || ' => ' || m.bot
            FROM memory m JOIN vector_map vm ON m.id = vm.memory_id
            WHERE vm.vector_id IN ({placeholders})
            """,
            indices,
        )
        return [row[0] for row in cursor.fetchall()]

    def fetch_by_ids(self, memory_ids: List[int]) -> List[Dict[str, Any]]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" * len(memory_ids))
        cursor = self.conn.execute(
            f"SELECT id, user, bot, important, timestamp FROM memory WHERE id IN ({placeholders})",
            memory_ids,
        )
        return [
            {
                "id": r[0],
                "user": r[1],
                "bot": r[2],
                "important": bool(r[3]),
                "timestamp": r[4],
            }
            for r in cursor
        ]

    def mark_important(self, memory_id: int) -> None:
        self.conn.execute("UPDATE memory SET important = 1 WHERE id = ?", (memory_id,))
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT id, user, bot, important, timestamp
            FROM memory ORDER BY timestamp DESC LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "id": r[0],
                "user": r[1],
                "bot": r[2],
                "important": bool(r[3]),
                "timestamp": r[4],
            }
            for r in cursor
        ]

    def close(self) -> None:
        self.conn.close()

    # ---------- внутренние ----------

    def _encode(self, text: str) -> Optional[bytes]:
        if not self.embedder:
            return None
        try:
            vec = self.embedder.encode(text, convert_to_numpy=True)
            faiss.normalize_L2(vec.reshape(1, -1))
            return vec.tobytes()
        except Exception as e:
            logger.error("Failed to create embedding: %s", e)
            return None

    def _check_limits(self) -> None:
        count = self.count()
        if count <= self.max_records:
            return

        to_delete = count - self.max_records
        cursor = self.conn.execute(
            "SELECT id FROM memory WHERE important = 0 ORDER BY timestamp ASC LIMIT ?",
            (to_delete,),
        )
        ids = [r[0] for r in cursor]
        if not ids:
            return

        placeholders = ",".join("?" * len(ids))
        vec_ids = []
        if self.vector_search:
            cursor = self.conn.execute(
                f"SELECT vector_id FROM vector_map WHERE memory_id IN ({placeholders})",
                ids,
            )
            vec_ids = [r[0] for r in cursor]

        self.conn.execute(f"DELETE FROM memory WHERE id IN ({placeholders})", ids)
        self.conn.execute(f"DELETE FROM vector_map WHERE memory_id IN ({placeholders})", ids)

        if self.vector_search and len(vec_ids) > 100:
            self._rebuild_index()

        self.conn.commit()
        logger.info("Cleaned up %d old records", len(ids))

    def _rebuild_index(self) -> None:
        if not self.vector_search:
            return
        logger.info("Rebuilding FAISS index...")
        new_index = faiss.IndexFlatIP(self.embedding_dim)
        cursor = self.conn.execute(
            "SELECT m.id, m.embedding FROM memory m WHERE m.embedding IS NOT NULL ORDER BY m.id"
        )
        self.conn.execute("DELETE FROM vector_map")

        vector_id = 0
        for memory_id, blob in cursor:
            if not blob:
                continue
            vec = np.frombuffer(blob, dtype=np.float32).reshape(1, -1)
            new_index.add(vec)
            self.conn.execute(
                "INSERT INTO vector_map (memory_id, vector_id) VALUES (?, ?)",
                (memory_id, vector_id),
            )
            vector_id += 1

        self.conn.commit()
        self.index = new_index
        faiss.write_index(self.index, self.idx_file)
        logger.info("Rebuilt index with %d vectors", self.index.ntotal)
