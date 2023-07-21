FROM tiangolo/meinheld-gunicorn-flask:python3.9
#WORKDIR /root/

#For OpenCV
RUN apt update
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

#Requirements
COPY docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./app /app

#Set prefix URL for production
ENV PREFIX="/read_serv"