FROM python:3.8-slim-buster
RUN mkdir ./LicensePlateApp
WORKDIR ./LicensePlateApp

RUN python3 -m venv /opt/venv
RUN . /opt/venv/bin/activate

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD [ "python3", "main.py"]


