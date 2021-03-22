# Ubuntu as base image
FROM python:3.7

# # Setting up the enviroment
RUN apt-get update --fix-missing
RUN apt install -y wget
# RUN apt install -y python3.7.4
RUN apt-get install -y python3-pip


# Defining a working enviroment
WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r requirements.txt

CMD ["python", "main_func.py"] 