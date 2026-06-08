FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential cmake pkg-config \
    libgl1-mesa-glx libegl1-mesa libosmesa6-dev libglfw3 libglew-dev \
    libgl1-mesa-dev libxrender1 libxext6 libsm6 ffmpeg \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/libero-venv
ENV PATH=/opt/libero-venv/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel

# Pin a modern-but-not-latest torch stack compatible with CUDA 11.8.
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

WORKDIR /workspace
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /workspace/LIBERO
WORKDIR /workspace/LIBERO

# LIBERO requirements may pull simulator dependencies. Keep this layer isolated.
RUN pip install -r requirements.txt || true
RUN pip install -e .

ENV PYTHONPATH=/workspace/LIBERO:$PYTHONPATH

CMD ["bash"]
