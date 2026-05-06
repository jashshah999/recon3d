FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    ffmpeg libgl1-mesa-glx libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY recon3d/ recon3d/

RUN pip install --no-cache-dir ".[all]"

ENTRYPOINT ["recon3d"]
CMD ["--help"]
