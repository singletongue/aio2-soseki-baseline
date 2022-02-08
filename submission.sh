#!/bin/bash
INPUT_FILE=$1
OUTPUT_FILE=$2

ONNX_MODEL_DIR="models/onnx"
FAISS_INDEX_FILE="models/passage_embeddings.idx"
PASSAGE_FILE="models/passages.tsv.gz"

python predict.py \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --onnx_model_dir $ONNX_MODEL_DIR \
    --passage_embeddings_file $FAISS_INDEX_FILE \
    --passage_file $PASSAGE_FILE
