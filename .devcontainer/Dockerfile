FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libomp-dev \
    libboost-all-dev \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user
RUN useradd -m vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER vscode
WORKDIR /workspace

RUN pip3 install --upgrade pip