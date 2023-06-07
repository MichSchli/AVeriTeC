import argparse
from pathlib import Path
import os
import json
import sys

parser = argparse.ArgumentParser(description='Combine averitec files split according to the combine_decorate_files scheme.')
parser.add_argument('--scheme', default="/rds/user/mss84/hpc-work/datasets/averitec/full_data/date.test.with_qs.X.json", help='')
args = parser.parse_args()

p = Path(args.scheme)
folder = p.parent
file_scheme = p.name
prefix, suffix = file_scheme.split("X")

examples = {}

for file in os.listdir(folder):
    if file.startswith(prefix) and file.endswith(suffix) and len(file.split(".")) == len(file_scheme.split(".")):
        span = [int(x) for x in file[len(prefix):-len(suffix)].split("-")]
        print("Reading examples " + str(span[0]) + " to " + str(span[1]) +  ".", file=sys.stderr)

        with open(str(folder) + "/" +file) as f:
            file_examples = json.load(f)
            for idx, example in enumerate(file_examples):
                if idx+span[0] in examples:
                    print("Error: Index overlap at index "+str(idx+span[0]), file=sys.stderr)
                    exit()
                examples[idx+span[0]] = example

print("Combining "+str(len(examples)) + " examples.", file=sys.stderr)
out_examples = [None] * len(examples)

for idx, example in examples.items():
    out_examples[idx] = example

for example in out_examples:
    if example is None:
        print("Error: Index mismatch", file=sys.stderr)
        exit()

print(json.dumps(out_examples, indent=4))