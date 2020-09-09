FROM python:3.8-slim
LABEL maintainer="Javier.Serrano@uab.cat"
# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive
# Update Ubuntu Software repository
WORKDIR /usr/src/app
ADD ./ .
RUN apt update
# Install pip  from ubuntu repository
RUN apt install -y python3-pip wget software-properties-common && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean
#RUN pip install 
RUN python3 -m pip install -U pip
RUN pip install scipy==1.4.1
RUN pip install matplotlib
RUN pip install qibo
CMD ["bash"]
