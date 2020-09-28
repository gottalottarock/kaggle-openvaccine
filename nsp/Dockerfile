FROM ubuntu:16.04
RUn apt -y update; apt -y install wget
RUN apt -y install gcc g++ cmake
RUN wget https://codeload.github.com/hust220/nsp/tar.gz/1.7
RUN tar xvzf 1.7
RUN cd nsp-1.7 && bash install.sh
RUN apt -y install python3-pip
COPY run_nsp.py ./
COPY run_opt.py ./
