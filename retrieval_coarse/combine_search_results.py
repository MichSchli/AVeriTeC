import argparse
import pandas as pd
import os
import sys

parser = argparse.ArgumentParser(description='Combine downloaded search result tsv files.')
parser.add_argument('--pattern', default="/rds/user/mss84/hpc-work/datasets/averitec/full_data/search_test_vicuna/search.", help='')
args = parser.parse_args()

folder_name = os.path.dirname(args.pattern)
file_name = os.path.basename(args.pattern)

line = ["index", "claim", "link", "page", "search_string", "search_type", "store_file"]
line = "\t".join(line)
print(line)

for f in sorted(os.listdir(folder_name)):
    if f.startswith(file_name):
        print(f, file=sys.stderr)
        with open(folder_name + "/" + f) as url_file:
            first = True
            for line in url_file:
                if first:
                    first = False
                else:
                    if line.strip():
                        print(line.strip())