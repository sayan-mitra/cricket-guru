FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt streamlit

COPY backend backend
COPY frontend frontend
COPY data/wikipedia data/wikipedia
COPY data/gold data/gold

ENV PYTHONPATH=/app/backend
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0"]
