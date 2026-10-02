FROM python:3.10-slim
WORKDIR /root/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
EXPOSE 80
CMD ["python", "app.py"]
