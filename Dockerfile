FROM python:3.10.12

RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]