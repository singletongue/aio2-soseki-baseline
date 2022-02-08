import argparse
import json

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--onnx_model_dir", type=str)
parser.add_argument("--biencoder_ckpt_file", type=str)
parser.add_argument("--reader_ckpt_file", type=str)
parser.add_argument("--passage_embeddings_file", type=str, required=True)
parser.add_argument("--passage_db_file", type=str)
parser.add_argument("--passage_file", type=str)
parser.add_argument("--device", type=str)
args = parser.parse_args()

if args.onnx_model_dir is None and (args.biencoder_ckpt_file is None or args.biencoder_ckpt_file is None):
    raise ValueError("if --onnx_model_dir is unset, both of --biencoder_ckpt_file and --reader_ckpt_file must be set.")
if args.passage_embeddings_file is None:
    raise ValueError("--passage_embeddings_file must be set.")
if args.onnx_model_dir is None and (args.passage_db_file is None or args.passage_file is None):
    raise ValueError("--passage_db_file or --passage_file must be specified.")

if args.onnx_model_dir is not None:
    from soseki.end_to_end.onnx_modeling import OnnxEndToEndQuestionAnswering
    model = OnnxEndToEndQuestionAnswering(
        onnx_model_dir=args.onnx_model_dir,
        passage_embeddings_file=args.passage_embeddings_file,
        passage_db_file=args.passage_db_file,
        passage_file=args.passage_file,
    )
else:
    from soseki.end_to_end.modeling import EndToEndQuestionAnswering
    model = EndToEndQuestionAnswering(
        biencoder_ckpt_file=args.biencoder_ckpt_file,
        reader_ckpt_file=args.reader_ckpt_file,
        passage_embeddings_file=args.passage_embeddings_file,
        passage_db_file=args.passage_db_file,
        passage_file=args.passage_file,
        device=args.device,
    )

with open(args.input_file) as f, open(args.output_file, "w") as fo:
    for line in tqdm(f):
        input_item = json.loads(line)
        qid = input_item["qid"]
        question = input_item["question"]

        answer_candidates = model.answer_question(question, num_reading_passages=3)
        predicted_answer = answer_candidates[0].answer_text

        output_item = {"qid": qid, "prediction": predicted_answer}
        print(json.dumps(output_item, ensure_ascii=False), file=fo)
