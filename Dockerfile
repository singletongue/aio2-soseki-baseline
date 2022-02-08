FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
RUN pip install --no-cache-dir \
        faiss-cpu \
        lmdb \
        numpy \
        onnxruntime \
        tqdm \
        transformers[ja]

# Download transformers models in advance
ARG TRANSFORMERS_BASE_MODEL_NAME="cl-tohoku/bert-base-japanese-v2"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
ENV TRANSFORMERS_OFFLINE=1

# Copy files to the image
WORKDIR /app
COPY models/ models
COPY soseki/ soseki
COPY predict.py .
COPY submission.sh .
