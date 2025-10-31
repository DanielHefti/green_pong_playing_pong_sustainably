
FROM python:3.13.5-slim

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \ 
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "dqn.py"]