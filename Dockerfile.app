FROM python:3.12-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install -U setuptools pip
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./src/utils/ /app/utils/
COPY ./src/app.py /app/app.py

EXPOSE 80

ENTRYPOINT ["streamlit", "run", "/app/app.py", "--server.port=80", "--server.address=0.0.0.0"]