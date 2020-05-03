ARG BASE=nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    screen \
    htop \
    tmux \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app

RUN curl -so miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
  && bash miniconda.sh -b -p miniconda \
  && rm miniconda.sh

ENV PATH=/app/miniconda/bin:$PATH

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir torch==1.4.0 torchvision=0.5.0