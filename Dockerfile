FROM python:3.11-slim

# libgl1/libglib for opencv, libzbar0 for pyzbar
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core ./core
COPY api ./api
COPY app.py cli.py ./

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
