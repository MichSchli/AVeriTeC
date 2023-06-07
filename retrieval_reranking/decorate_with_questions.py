import argparse
import json
import pandas as pd
import nltk
from rank_bm25 import BM25Okapi
import numpy as np
import os
import sys
import torch
import tqdm
from transformers import BloomTokenizerFast, BloomModel, BloomForCausalLM
from accelerate import Accelerator


parser = argparse.ArgumentParser(description='Use a prompt to generate questions that could be answered by top-k retrieved evidence. Output generated questions.')
parser.add_argument('--reference_corpus', default="data/train.json", help='')
parser.add_argument('--target_file', default="data/dev.json", help='')
parser.add_argument('--url_file', default="search_results.tsv", help='')
parser.add_argument('--store_folder', default="store/retrieved_docs", help='')
parser.add_argument('--top_k', default=100, type=int, help='How many documents should we pick out with BM25')
parser.add_argument('--start_idx', default=0, type=int, help='Which claim to start at. Useful for larger corpus.')
parser.add_argument('--n_to_compute', default=-1, type=int, help='How many claims to work through. Useful for larger datasets.')
args = parser.parse_args()

# Construct the prompts to retrieve and set up BM25 to find similar evidence:

with open(args.reference_corpus) as f:
    train_examples = json.load(f)

def claim2prompts(example):
    claim = example["claim"]

    #claim_str = "Claim: " + claim + "||Evidence: "
    claim_str = "Evidence: "

    for question in example["questions"]:
        q_text = question["question"].strip()
        if len(q_text) == 0:
            continue

        if not q_text[-1] == "?":
            q_text += "?"

        answer_strings = []

        for a in question["answers"]:
            if a["answer_type"] in ["Extractive", "Abstractive"]:
                answer_strings.append(a["answer"])
            if a["answer_type"] == "Boolean":
                answer_strings.append(a["answer"]  + ", because " + a["boolean_explanation"].lower().strip())

        for a_text in answer_strings:
            if not a_text[-1] in [".", "!", ":", "?"]:
                a_text += "."

            #prompt_lookup_str = claim + " " + a_text
            prompt_lookup_str = a_text
            this_q_claim_str = claim_str + " " + a_text.strip() + "||Question answered: " + q_text
            yield (prompt_lookup_str, this_q_claim_str.replace("\n", " ").replace("||", "\n"))

prompt_corpus = []
tokenized_corpus = []

for example in tqdm.tqdm(train_examples):
    for lookup_str, prompt in  claim2prompts(example):
        entry = nltk.word_tokenize(lookup_str)
        tokenized_corpus.append(entry)
        prompt_corpus.append(prompt)
    
prompt_bm25 = BM25Okapi(tokenized_corpus)


# Attach evidence to examples:

with open(args.target_file) as f:
    examples = json.load(f)

with open(args.url_file) as url_file:
    first = True
    for line in url_file:
        if first:
            first = False
        else:
            line_parts = line.strip().split("\t")

            if len(line_parts) < 7:
                continue
            idx = int(line_parts[0])

            #claim = line_parts[1]
            url = line_parts[2]
            store_file = args.store_folder + "/" + line_parts[6].split("/")[-1]

            if "retrieved_store_files" not in examples[idx]:
                examples[idx]["retrieved_store_files"] = []
            examples[idx]["retrieved_store_files"].append(store_file)

# Define the bloom model:
accelerator = Accelerator()
accel_device = accelerator.device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-7b1")
model = BloomForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    offload_folder="./offload"
)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Go through the dataset, generating questions for the evidence: 
end_idx = -1
if args.n_to_compute != -1:
    end_idx = args.start_idx+args.n_to_compute
    
for idx,example in enumerate(tqdm.tqdm(examples[args.start_idx:end_idx])):
    # First, retrieve top 50 documents with bm25:
    tokenized_corpus = []
    all_data_corpus = []

    this_example_store_files = [] if "retrieved_store_files" not in example else example["retrieved_store_files"] 
    for store_file in this_example_store_files:
        with open(store_file, 'r') as f:
            first = True
            for line in f:
                line = line.strip()

                if first:
                    first = False
                    location_url = line
                    continue
                
                if len(line) > 3:
                    entry = nltk.word_tokenize(line)
                    if (location_url, line) not in all_data_corpus:
                        tokenized_corpus.append(entry)
                        all_data_corpus.append((location_url, line))

    if len(tokenized_corpus) == 0:
        print("")
        continue
    
    bm25 = BM25Okapi(tokenized_corpus)
    s = bm25.get_scores(nltk.word_tokenize(example["claim"]))
    n_coarse = args.top_k
    top_n = np.argsort(s)[::-1][:n_coarse]
    docs = [all_data_corpus[i] for i in top_n]

    tracker = []
    docs_with_qs = []

    # Then, generate questions for those top 50:
    for doc in docs:
        #prompt_lookup_str = example["claim"] + " " + doc[1]
        prompt_lookup_str = doc[1]

        prompt_s = prompt_bm25.get_scores(nltk.word_tokenize(prompt_lookup_str))
        prompt_n = 10
        prompt_top_n = np.argsort(prompt_s)[::-1][:prompt_n]
        prompt_docs = [prompt_corpus[i] for i in prompt_top_n]

        claim_prompt = "Evidence: " + doc[1].replace("\n", " ") + "\nQuestion answered: "

        prompt = "\n\n".join(prompt_docs + [claim_prompt])

        sentences = [prompt]

        inputs = tokenizer(
        sentences, 
        padding=True,
        return_tensors="pt").to(device)

        outputs = model.generate(inputs["input_ids"],
            max_length=5000, 
            num_beams=2, 
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        tgt_text = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]

        # We are not allowed to generate more than 250 characters:
        tgt_text = tgt_text[:250]
        
        qa_pair = [tgt_text.strip().split("?")[0].replace("\n", " ") + "?",  doc[1].replace("\n", " "), doc[0]]

        if not "bm25_qas" in example:
            example["bm25_qas"] = []

        example["bm25_qas"].append(qa_pair)

print(json.dumps(examples[args.start_idx:args.start_idx+args.n_to_compute], indent=4))