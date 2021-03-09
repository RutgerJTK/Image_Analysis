# Ubuntu as base image
FROM ubuntu

# Setting up the enviroment
RUN apt-get update --fix-missing
RUN apt install -y wget
RUN apt install -y python3.8

# Defining a working enviroment
WORKDIR /app

