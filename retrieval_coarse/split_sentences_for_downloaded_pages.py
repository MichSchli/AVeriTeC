import argparse
import os
import tqdm

from html2lines import line_correction

parser = argparse.ArgumentParser(description='Do proper sentence splitting for all downloaded pages.')
parser.add_argument('--store_folder', default="/rds/user/mss84/hpc-work/datasets/averitec/full_data/retrieved_docs", help='')
parser.add_argument('--start_idx', default=0, type=int, help='Which claim to start at. Useful for larger corpus.')
parser.add_argument('--n_to_compute', default=50000, type=int, help='How many claims to work through. Useful for larger corpus.')
args = parser.parse_args()

if not os.path.exists(args.store_folder + ".formatted"):
    os.makedirs(args.store_folder + ".formatted")

end_idx = -1
if args.n_to_compute != -1:
    end_idx = args.start_idx+args.n_to_compute

for file in tqdm.tqdm(sorted(list(os.listdir(args.store_folder)))[args.start_idx:end_idx]):
    full_path = args.store_folder + "/" + file

    lines = []
    with open(full_path, "r") as f:
        for l in f:
            lines.append(l.strip())

    fixed_lines = line_correction(lines)

    formatted_path = args.store_folder + ".formatted" + "/" + file

    with open(formatted_path, "w") as out_f:
        print("\n".join(fixed_lines), file=out_f)   