FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/saber/src

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.10 python3-pip git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/saber
COPY requirements.txt pyproject.toml README.md LICENSE NOTICE ./
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN python3 -m pip install --no-cache-dir --no-deps -e .

CMD ["python3", "-m", "saber.cli.verify"]
