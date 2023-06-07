import numpy as np
import argparse
import json
import nltk
from rank_bm25 import BM25Okapi
import tqdm
import torch
from transformers import BloomTokenizerFast, BloomModel, BloomForCausalLM
from accelerate import Accelerator

parser = argparse.ArgumentParser(description='Generate extra questions based on claims with a prompt. Useful for searching.')
parser.add_argument('--reference_corpus', default="data/train.json", help='')
parser.add_argument('--target_file', default="data/dev.json", help='')
parser.add_argument('--n', default=10, help='')
args = parser.parse_args()

with open(args.target_file) as f:
    j = json.load(f)
    examples = j #["examples"]

with open(args.reference_corpus) as f:
    j = json.load(f)
    train_examples = j

all_data_corpus = []
tokenized_corpus = []

for train_example in train_examples:
    train_claim = train_example["claim"]

    speaker = train_example["speaker"].strip() if train_example["speaker"] is not None and len(train_example["speaker"]) > 1 else "they"

    questions = [q["question"] for q in train_example["questions"]]

    claim_dict_builder = {}
    claim_dict_builder["claim"] = train_claim
    claim_dict_builder["speaker"] = speaker
    claim_dict_builder["questions"] = questions

    tokenized_corpus.append(nltk.word_tokenize(claim_dict_builder["claim"]))
    all_data_corpus.append(claim_dict_builder)

bm25 = BM25Okapi(tokenized_corpus)

# Define methods to transform retrieved docs into a prompt:
def doc2prompt(doc):
    prompt_parts = "Outrageously, " + doc["speaker"] + " claimed that \"" + doc["claim"].strip() + "\". Criticism includes questions like: "
    questions = [q.strip() for q in doc["questions"]]
    return prompt_parts + " ".join(questions)

def docs2prompt(top_docs):
    return "\n\n".join([doc2prompt(d) for d in top_docs])

# Define the bloom model:
accelerator = Accelerator()
accel_device = accelerator.device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-7b1")
model = BloomForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    torch_dtype=torch.bfloat16,
).to(device)

for example in tqdm.tqdm(examples):
    test_claim = example["claim"]
    speaker = example["speaker"].strip() if example["speaker"] is not None and len(example["speaker"]) > 1 else "they"

    s = bm25.get_scores(nltk.word_tokenize(test_claim))
    top_n = np.argsort(s)[::-1][:args.n]
    docs = [all_data_corpus[i] for i in top_n]

    prompt = docs2prompt(docs) + "\n\n" + "Outrageously, " + speaker + " claimed that \""+ test_claim.strip() + "\". Criticism includes questions like: "
    sentences = [prompt]

    inputs = tokenizer(
    sentences, 
    padding=True,
    return_tensors="pt").to(device)

    outputs = model.generate(inputs["input_ids"],
        max_length=2000, 
        num_beams=2, 
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    tgt_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    in_len = len(sentences[0])
    questions_str = tgt_text[in_len:].split("\n")[0]

    qs = questions_str.split("?")
    qs = [q.strip() + "?" for q in qs if q.strip() and len(q.strip()) < 300]

    example["questions"] = [{"question": q, "answers": []} for q in qs]

print(json.dumps(examples, indent=4))