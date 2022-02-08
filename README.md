# AIO2 BPR Baseline

This is a question answering (QA) system based on [studio-ousia/soseki](https://github.com/studio-ousia/soseki), which utilizes [Binary Passage Retriever (BPR)](https://github.com/studio-ousia/bpr), an efficient passages retrieval model for a large collection of documents.

This work is provided as one of the baseline systems for [AIO2 competition](https://sites.google.com/view/project-aio/competition2).

## Installation

You can install the required libraries using pip:

```sh
$ pip install -r requirements.txt
```

**Note:** If you are using a GPU Environment different from CUDA 10.2, you may need to reinstall PyTorch according to [the official documentation](https://pytorch.org/get-started/locally/).

## Example Usage

Before you start, you need to download the datasets available at
[cl-tohoku/AIO2_DPR_baseline](https://github.com/cl-tohoku/AIO2_DPR_baseline) by running the downloading script `scripts/download_data.sh`.

```sh
$ bash scripts/download_data.sh <DATASET_DIR>
```

In the following experiments, we used a server with 4 GeForce RTX 2080 GPUs with 11GB memory.

**1. Build passage database**

```sh
$ python build_passage_db.py \
    --passage_file <DATASET_DIR>/wiki/jawiki-20210503-paragraphs.tsv.gz \
    --db_file <WORK_DIR>/passages.db \
    --db_map_size 10000000000 \
    --skip_header
```

**2. Train a biencoder**

```sh
$ python train_biencoder.py \
    --train_file <DATASET_DIR>/aio/abc_01-12_retriever.json.gz \
    --dev_file <DATASET_DIR>/aio/aio_01_dev_retriever.json.gz \
    --output_dir <WORK_DIR>/biencoder \
    --max_question_length 128 \
    --max_passage_length 256 \
    --num_negative_passages 1 \
    --shuffle_hard_negative_passages \
    --shuffle_normal_negative_passages \
    --base_pretrained_model cl-tohoku/bert-base-japanese-v2 \
    --binary \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 1e-5 \
    --warmup_proportion 0.1 \
    --gradient_clip_val 2.0 \
    --max_epochs 20 \
    --gpus 4 \
    --precision 16 \
    --accelerator ddp
```

**3. Build passage embeddings**

```sh
$ python build_passage_embeddings.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --output_file <WORK_DIR>/passage_embeddings.idx \
    --max_passage_length 256 \
    --batch_size 2048 \
    --device_ids 0,1,2,3
```

**4. Evaluate the retriever and create datasets for reader**

```sh
$ mkdir <WORK_DIR>/reader_data

$ python evaluate_retriever.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --passage_embeddings_file <WORK_DIR>/passage_embeddings.idx \
    --qa_file <DATASET_DIR>/aio/abc_01-12_retriever.tsv \
    --output_file <WORK_DIR>/reader_data/abc_01-12.jsonl \
    --batch_size 64 \
    --max_question_length 128 \
    --top_k 1,2,5,10,20,50,100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type simple_nfkc \
    --include_title_in_passage \
    --device_ids 0,1,2,3
# The result should be logged as follows:
# Recall at 1: 0.6041 (10714/17735)
# Recall at 2: 0.7123 (12633/17735)
# Recall at 5: 0.8030 (14242/17735)
# Recall at 10: 0.8415 (14924/17735)
# Recall at 20: 0.8686 (15404/17735)
# Recall at 50: 0.8937 (15849/17735)
# Recall at 100: 0.9064 (16075/17735)

$ python evaluate_retriever.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --passage_embeddings_file <WORK_DIR>/passage_embeddings.idx \
    --qa_file <DATASET_DIR>/aio/aio_01_dev_retriever.tsv \
    --output_file <WORK_DIR>/reader_data/aio_01_dev.jsonl \
    --batch_size 64 \
    --max_question_length 128 \
    --top_k 1,2,5,10,20,50,100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type simple_nfkc \
    --include_title_in_passage \
    --device_ids 0,1,2,3
# The result should be logged as follows:
# Recall at 1: 0.6160 (1227/1992)
# Recall at 2: 0.7279 (1450/1992)
# Recall at 5: 0.8308 (1655/1992)
# Recall at 10: 0.8740 (1741/1992)
# Recall at 20: 0.9096 (1812/1992)
# Recall at 50: 0.9458 (1884/1992)
# Recall at 100: 0.9639 (1920/1992)

$ python evaluate_retriever.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --passage_embeddings_file <WORK_DIR>/passage_embeddings.idx \
    --qa_file <DATASET_DIR>/aio/aio_01_test_retriever.tsv \
    --output_file <WORK_DIR>/reader_data/aio_01_test.jsonl \
    --batch_size 64 \
    --max_question_length 128 \
    --top_k 1,2,5,10,20,50,100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type simple_nfkc \
    --include_title_in_passage \
    --device_ids 0,1,2,3
# The result should be logged as follows:
# Recall at 1: 0.5875 (1175/2000)
# Recall at 2: 0.7055 (1411/2000)
# Recall at 5: 0.8140 (1628/2000)
# Recall at 10: 0.8675 (1735/2000)
# Recall at 20: 0.9020 (1804/2000)
# Recall at 50: 0.9370 (1874/2000)
# Recall at 100: 0.9580 (1916/2000)
```

**5. Train a reader**

```sh
$ python train_reader.py \
    --train_file <WORK_DIR>/reader_data/abc_01-12.jsonl \
    --dev_file <WORK_DIR>/reader_data/aio_01_dev.jsonl \
    --output_dir <WORK_DIR>/reader \
    --train_num_passages 16 \
    --eval_num_passages 50 \
    --max_input_length 384 \
    --include_title_in_passage \
    --shuffle_positive_passage \
    --shuffle_negative_passage \
    --num_dataloader_workers 1 \
    --base_pretrained_model cl-tohoku/bert-base-japanese-v2 \
    --answer_normalization_type simple_nfkc \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --learning_rate 1e-5 \
    --warmup_proportion 0.1 \
    --accumulate_grad_batches 4 \
    --gradient_clip_val 2.0 \
    --max_epochs 10 \
    --gpus 4 \
    --precision 16 \
    --accelerator ddp
```

**6. Evaluate the reader**

```sh
$ python evaluate_reader.py \
    --reader_file <WORK_DIR>/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --test_file <WORK_DIR>/reader_data/aio_01_dev.jsonl \
    --test_num_passages 100 \
    --test_max_load_passages 100 \
    --test_batch_size 4 \
    --gpus 4 \
    --accelerator ddp
# The result should be printed as follows:
# --------------------------------------------------------------------------------
# DATALOADER:0 TEST RESULTS
# {'test_answer_accuracy': 0.6882529854774475,
#  'test_classifier_precision': 0.8012048006057739}
# --------------------------------------------------------------------------------
```

```sh
$ python evaluate_reader.py \
    --reader_file <WORK_DIR>/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --test_file <WORK_DIR>/reader_data/aio_01_test.jsonl \
    --test_num_passages 100 \
    --test_max_load_passages 100 \
    --test_batch_size 4 \
    --gpus 4 \
    --accelerator ddp
# The result should be printed as follows:
# --------------------------------------------------------------------------------
# DATALOADER:0 TEST RESULTS
# {'test_answer_accuracy': 0.6915000081062317,
#  'test_classifier_precision': 0.8005000352859497}
# --------------------------------------------------------------------------------
```

**7. Convert the trained models into ONNX format**

```sh
$ python convert_models_to_onnx.py \
    --biencoder_ckpt_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --reader_ckpt_file <WORK_DIR>/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --output_dir <WORK_DIR>/onnx
```

### Subission for AIO2 Competition

**1. Copy the models and data files into `models/` directory**

The files should be placed and renamed as follows:

```sh
models
├── passage_embeddings.idx      # from <WORK_DIR>/passage_embeddings.idx
├── onnx
│   ├── biencoder_hparams.json  # from <WORK_DIR>/onnx/biencoder_hparams.json
│   ├── question_encoder.onnx   # from <WORK_DIR>/onnx/question_encoder.onnx
│   ├── reader.onnx             # from <WORK_DIR>/onnx/reader.onnx
│   └── reader_hparams.json     # from <WORK_DIR>/onnx/reader_hparams.json
└── passages.tsv.gz             # from <DATASET_DIR>/wiki/jawiki-20210503-paragraphs.tsv.gz
```

**Note:** You do not have to include `<WORK_DIR>/onnx/passage_encoder.onnx` in the directory since it is not used in the prediction stage.

**2. Build the Docker image**

```sh
$ docker build -t aio2-bpr-baseline .
```

You can find the size of the image by executing the command below:

```sh
$ docker run --rm aio2-bpr-baseline du -h --max-depth=0 /
```

**3. Run the image to perform prediction**

We assume `<TEST_DATA_DIR>` contains a test file such as [`aio_02_dev_unlabeled_v1.0.jsonl`](https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_dev_unlabeled_v1.0.jsonl), which is distributed on the [AIO2 official website](https://sites.google.com/view/project-aio/competition2).

Be sure to specify `<TEST_DATA_DIR>` by the absolute path.

```sh
$ docker run --rm -v <TEST_DATA_DIR>:/app/data -it aio2-bpr-baseline \
    bash submission.sh data/aio_02_dev_unlabeled_v1.0.jsonl data/predictions.jsonl
```

The prediction result will be saved to `<TEST_DATA_DIR>/predictions.jsonl`.

**4. Save a Docker image to file**

```sh
$ docker save aio2-bpr-baseline | gzip > aio2-bpr-baseline.tar.gz
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This
work is licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative
Commons Attribution-NonCommercial 4.0 International License</a>.
