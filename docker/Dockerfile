FROM ubuntu:22.04

RUN apt update -y && \
    apt install -y \
    hping3 \
    inetutils-ping \
    netcat \
    nmap \
    telnet \
    tshark

COPY server /
COPY client /

ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]
