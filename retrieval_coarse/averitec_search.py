import argparse
from time import sleep
import pandas as pd
import tqdm
from googleapiclient.discovery import build
import argparse
import json
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import requests
import io
from bs4 import BeautifulSoup
from html2lines import url2lines
from nltk import pos_tag, word_tokenize
import threading
import gc
import os

parser = argparse.ArgumentParser(description='Download and store search pages for FCB files.')
parser.add_argument('--averitec_file', default="data/dev.generated_questions.json", help='')
parser.add_argument('--misinfo_file', default="data/misinfo_list.txt", help='')
parser.add_argument('--n_pages', default=3, help='')
parser.add_argument('--store_folder', default="store/retrieved_docs", help='')
parser.add_argument('--start_idx', default=0, type=int, help='Which claim to start at. Useful for larger corpus.')
parser.add_argument('--n_to_compute', default=-1, type=int, help='How many claims to work through. Useful for larger corpus.')
parser.add_argument('--resume', default="", help='Resume work from a particular file. Useful for larger corpus.')
args = parser.parse_args()

existing = {}
first = True
if args.resume != "":
    next_claim = {"claim": None}
    for line in open(args.resume, "r"):
        # skip the first line
        if first:
            first = False
            continue

        parts = line.strip().split("\t")
        claim = parts[1]

        if claim != next_claim["claim"]: # Bit of a hack but I am intertionally causing a fencepost error here to rebuild the last claim, as we do not know if it was finished or not
            if next_claim["claim"] is not None:
                existing[next_claim["claim"]] = next_claim
            next_claim = {"claim": claim, "lines": []}

        next_claim["lines"].append(line.strip())

if not os.path.exists(args.store_folder):
    os.makedirs(args.store_folder)

api_key = "YOUR_GOOGLE_CSE_API_KEY"
search_engine_id = "YOUR_CSE_ID"

start_idx = 0
misinfo_list_file = args.misinfo_file
misinfo_list = []

blacklist = [
    "jstor.org", # Blacklisted because their pdfs are not labelled as such, and clog up the download
    "facebook.com", # Blacklisted because only post titles can be scraped, but the scraper doesn't know this,
    "ftp.cs.princeton.edu", # Blacklisted because it hosts many large NLP corpora that keep showing up
    "nlp.cs.princeton.edu",
    "huggingface.co"
]

blacklist_files = [ # Blacklisted some NLP nonsense that crashes my machine with OOM errors
    "/glove.", 
    "ftp://ftp.cs.princeton.edu/pub/cs226/autocomplete/words-333333.txt",
    "https://web.mit.edu/adamrose/Public/googlelist",
]


for line in open(misinfo_list_file, "r"):
    if line.strip():
        misinfo_list.append(line.strip().lower())

def get_domain_name(url):
    if '://' not in url:
        url = 'http://' + url

    domain = urlparse(url).netloc

    if domain.startswith("www."):
        return domain[4:]
    else:
        return domain

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()

    if "items" in res:
        return res['items']
    else:
        return []

pages = 0
a_pages = 0
found_pages = 0
found_pages_1hop = 0
n_pages = 0

with open(args.averitec_file) as f:
    j = json.load(f)
    examples = j

def string_to_search_query(text, author):
    parts = word_tokenize(text.strip())
    tags = pos_tag(parts)

    keep_tags = ["CD", "JJ", "NN", "VB"]

    if author is not None:
        search_string = author.split()
    else:
        search_string = []

    for token, tag in zip(parts, tags):
        for keep_tag in keep_tags:
            if tag[1].startswith(keep_tag):
                search_string.append(token)

    search_string = " ".join(search_string)
    return search_string

def get_google_search_results(api_key, search_engine_id, google_search, sort_date, search_string, page=0):
    search_results = []
    for i in range(3):
        try:
            search_results += google_search(
            search_string,
            api_key,
            search_engine_id,
            num=10,
            start=0 + 10 * page,
            sort="date:r:19000101:"+sort_date,
            dateRestrict=None,
            gl="US"
            )
            break
        except:
            sleep(3)

    return search_results

def get_and_store(url_link, fp, worker):
    page_lines = url2lines(url_link)

    with open(fp, "w") as out_f:
        print("\n".join([url_link] + page_lines), file=out_f)   

    worker_stack.append(worker)  
    gc.collect()

line = ["index", "claim", "link", "page", "search_string", "search_type", "store_file"]
line = "\t".join(line)
print(line)

worker_stack = list(range(10))

end_idx = -1
if args.n_to_compute != -1:
    end_idx = args.start_idx+args.n_to_compute

index = args.start_idx -1
for _, example in tqdm.tqdm(list(enumerate(examples[args.start_idx:end_idx]))):
    index += 1
    claim = example["claim"]

    # If we already have this claim in the file we are resuming from, skip it
    if claim in existing:
        for line in existing[claim]["lines"]:
            print(line)
        continue

    speaker = example["speaker"].strip() if example["speaker"] else None

    questions = [q["question"] for q in example["questions"]]

    try:
        year, month, date = example["check_date"].split("-")
    except:
        month, date, year = "01", "01", "2022"
    
    if len(year) == 2 and int(year) <= 30:
        year = "20" + year
    elif len(year) == 2:
        year = "19" + year
    elif len(year) == 1:
        year = "200" + year

    if len(month) == 1:
        month = "0" + month
    
    if len(date) == 1:
        date = "0" + date

    sort_date = year + month + date
    
    search_strings = []
    search_types = []

    if speaker is not None:
        search_string = string_to_search_query(claim, speaker)
        search_strings.append(search_string)
        search_types += ["claim+author"]

    search_string_2 = string_to_search_query(claim, None)
    
    search_strings += [
        search_string_2,
        claim,
        ]

    search_types += [
        "claim",
        "claim-noformat",
        ]

    search_strings += questions
    search_types += ["question" for _ in questions]

    search_results = []
    visited = {}
    
    store_counter = 0
    ts = []
    for this_search_string, this_search_type in zip(search_strings, search_types):
        for page_num in range(args.n_pages):
            search_results = get_google_search_results(api_key, search_engine_id, google_search, sort_date, this_search_string, page=page_num)

            for result in search_results:
                link = str(result["link"])

                domain = get_domain_name(link)

                if domain in blacklist:
                    continue

                broken = False
                for b_file in blacklist_files:
                    if b_file in link:
                        broken = True
        
                if broken:
                    continue

                if link.endswith(".pdf") or link.endswith(".doc"):
                    continue

                store_file_path = ""

                if link in visited:
                    store_file_path = visited[link]
                else:
                    store_counter += 1

                    store_file_path = args.store_folder + "/search_result_" + str(index) + "_" + str(store_counter) + ".store"
                    visited[link] = store_file_path 

                    while len(worker_stack) == 0: # Wait for a wrrker to become available. Check every second.
                        sleep(1)

                    worker = worker_stack.pop()

                    t = threading.Thread(target=get_and_store, args=(link, store_file_path, worker))
                    t.start()
                            
    
                line = [str(index), claim, link, str(page_num), this_search_string, this_search_type, store_file_path]
                line = "\t".join(line)
                print(line)