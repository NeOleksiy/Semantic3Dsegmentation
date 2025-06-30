FROM nvcr.io/nvidia/pytorch:23.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenmpi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    openssh-client \
    git \
    wget \
    unzip \
    ffmpeg \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

ENV PYTHONPATH=/workspace:$PYTHONPATH

RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r requirements.txt && \
    pip install hydra-core open3d py3dmol

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]