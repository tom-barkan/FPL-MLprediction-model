FROM python:3.11-slim

WORKDIR /app

# Cache bust: v2
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 8501

ENTRYPOINT ["./start.sh"]
