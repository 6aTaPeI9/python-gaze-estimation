FROM centos:7

USER root
ENV SHELL /bin/bash

WORKDIR /usr/bin/
RUN yum -y install epel-release
RUN yum -y update
RUN yum -y groupinstall "Development Tools"
RUN yum -y install openssl-devel bzip2-devel libffi-devel xz-devel
RUN yum -y install wget
RUN wget https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tgz
RUN tar xvf Python-3.8.16.tgz
WORKDIR /usr/bin/Python-3.8.16/
RUN ./configure --enable-optimizations
RUN make altinstall

RUN yum -y install mesa-libGL
RUN yum -y install libSM.so.6
RUN yum -y install libXext.so.6
RUN yum -y install libXrender.so.1
RUN yum -y install libfontconfig.so.1

RUN pip3.8 install --upgrade pip
RUN pip3.8 install cmake
RUN yum -y install cmake
COPY . /home/gaze_tracking
WORKDIR /home/gaze_tracking
RUN pip3.8 install -r ./requirements.txt
ENTRYPOINT ["/bin/sh","-c","sleep infinity"]