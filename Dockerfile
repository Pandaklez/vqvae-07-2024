FROM nvcr.io/nvidia/pytorch:22.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ENV_NAME=vqvae-anya
ENV PYTHON_VERSION=3.9
WORKDIR /workspace/project

RUN apt-get update
RUN apt-get install rename
RUN apt-get install -y ffmpeg

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /conda \
    && echo export PATH=/conda/bin:$PATH >> .bashrc \
    && rm miniconda3.sh
ENV PATH="/conda/bin:${PATH}"
RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}


# Switch to bash shell
SHELL ["/bin/bash", "-c"]


# Install requirements
COPY requirements.txt ./
RUN source activate ${CONDA_ENV_NAME} \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt


# Set ${CONDA_ENV_NAME} to default virutal environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

