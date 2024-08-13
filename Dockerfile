FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y xvfb


RUN pip install opencv-python-headless \
    JIMG==2.1.4


CMD ["python3", "-c", "from JIMG.app.load_app import run; run()"]
