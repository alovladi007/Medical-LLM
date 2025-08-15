FROM python:3.11-slim
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app /app
EXPOSE 8006
CMD ["uvicorn","imaging_api:app","--host","0.0.0.0","--port","8006"]
