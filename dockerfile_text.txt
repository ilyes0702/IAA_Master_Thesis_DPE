FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Added sudo, gfortran, and the BLAS/LAPACK/SUNDIALS libraries
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libomp-dev \
    libboost-all-dev \
    git \
    build-essential \
    curl \
    sudo \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libsundials-dev \
    texlive-latex-extra \
    texlive-fonts-recommended \
    dvipng \
    ghostscript \
    cm-super \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user and give them sudo privileges
RUN useradd -m vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER vscode
WORKDIR /workspace

RUN pip3 install --upgrade pip