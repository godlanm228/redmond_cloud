import hashlib
import os
from typing import Optional

def calculate_file_hash(filepath: str, algorithm: str = 'sha256') -> Optional[str]:
    """
    Вычислить хеш файла.

    Args:
        filepath: Путь к файлу
        algorithm: Алгоритм хеширования

    Returns:
        str: Хеш в hex формате или None
    """
    if not os.path.exists(filepath):
        return None

    hash_func = getattr(hashlib, algorithm)()

    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception:
        return None

def calculate_string_hash(data: str, algorithm: str = 'sha256') -> str:
    """Вычислить хеш строки"""
    hash_func = getattr(hashlib, algorithm)()
    hash_func.update(data.encode('utf-8'))
    return hash_func.hexdigest()