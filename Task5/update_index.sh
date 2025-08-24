#!/bin/bash
# Скрипт для автоматического обновления базы знаний

LOG_FILE="./update_index.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/update_index.py"

# Создаем папку для документов если не существует
mkdir -p ./knowledge_base

echo "$(date): Starting knowledge base update" >> "$LOG_FILE"

# Запускаем Python скрипт
python3 "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date): Update completed successfully" >> "$LOG_FILE"
else
    echo "$(date): Update failed with exit code $EXIT_CODE" >> "$LOG_FILE"
fi

exit $EXIT_CODE