#!/bin/bash

# Проверяем существование папок
if [ ! -d "raw_data" ]; then
    echo "Папка raw_data не существует!"
    exit 1
fi

if [ ! -d "knowledge_base" ]; then
    mkdir knowledge_base
fi

# Проверяем существование файла terms_map.json
if [ ! -f "terms_map.json" ]; then
    echo "Файл terms_map.json не существует!"
    exit 1
fi

# Создаем временный файл с sed-скриптом для замены
SED_SCRIPT=$(mktemp)

# Преобразуем JSON в sed-скрипт
jq -r 'to_entries[] | "s/\(.key)/\(.value)/gI"' terms_map.json > "$SED_SCRIPT"

# Обрабатываем каждый .md файл в папке raw_data
for file in raw_data/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        # Применяем замены и сохраняем результат в папку Data
        sed -f "$SED_SCRIPT" "$file" > "knowledge_base/$filename"
    fi
done

# Удаляем временный файл
rm "$SED_SCRIPT"

echo "Замена завершена. Результаты сохранены в папку knowledge_base."