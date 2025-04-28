FROM python:3.12-bookworm

ENV PYTHONUNBUFFERED=1

RUN apt update && apt install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install opencv-python numpy

RUN mkdir /app
RUN mkdir /app/input
RUN mkdir /app/output

WORKDIR /app
COPY ./img_align.py /app/img_align.py

CMD ["python3", "/app/img_align.py", "/app/input", "/app/output"]
