FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

COPY ./requirements.txt /app/requirements.txt
RUN pip install -U setuptools pip
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./src/utils/ /app/src/utils/
COPY ./src/app.py /app/src/app.py

EXPOSE 80

WORKDIR /app/src

CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]