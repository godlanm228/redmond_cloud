#!/usr/bin/env bash
# Ежедневный снимок базы хаба.
#
# VACUUM INTO даёт КОНСИСТЕНТНУЮ копию на работающем сервисе — в отличие от
# cp, который на WAL-базе может скопировать файл в середине транзакции.
# Останавливать бота не нужно.
#
# До 15.08.2026 бэкапов не было вообще: потеря data/ означала потерю всего —
# дневника, дедлайнов, целей, истории смен.
#
# Установка (раз в сутки в 04:30, время VM):
#   crontab -l | { cat; echo "30 4 * * * /home/ubuntu/redmond-hub/deploy/backup_db.sh"; } | crontab -
set -euo pipefail

HUB_DIR="${HUB_DIR:-$HOME/redmond-hub}"
DB="$HUB_DIR/data/memory.sqlite"
DEST_DIR="${BACKUP_DIR:-$HOME/backups/db}"
KEEP_DAYS="${KEEP_DAYS:-14}"

mkdir -p "$DEST_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
DEST="$DEST_DIR/hub-$STAMP.sqlite"

if [ ! -f "$DB" ]; then
    echo "backup_db: базы нет по пути $DB" >&2
    exit 1
fi

sqlite3 "$DB" "VACUUM INTO '$DEST'"
gzip -f "$DEST"

# Чистим старые снимки. -mtime считает СУТКИ, поэтому KEEP_DAYS=14 оставляет
# примерно две недели истории.
find "$DEST_DIR" -name 'hub-*.sqlite.gz' -mtime "+$KEEP_DAYS" -delete

echo "backup_db: $DEST.gz ($(du -h "$DEST.gz" | cut -f1)), снимков: $(ls -1 "$DEST_DIR" | wc -l)"
