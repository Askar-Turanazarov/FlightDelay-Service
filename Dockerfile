# Официальный легкий образ Python
FROM python:3.10-slim

# Устанавливаем рабочую папку в контейнере
WORKDIR /askar_app

# Копируем список зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Команда запуска
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]