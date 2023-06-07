from distutils.command.config import config
import requests
from time import sleep
import trafilatura
from trafilatura.meta import reset_caches
from trafilatura.settings import DEFAULT_CONFIG
import spacy
import sys
nlp = spacy.load('en_core_web_lg')
import sys

DEFAULT_CONFIG.MAX_FILE_SIZE = 50000

def get_page(url):
    page = None
    for i in range(3):
        try:
            page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)
            assert page is not None
            print("Fetched "+url, file=sys.stderr)
            break
        except:
            sleep(3)
    return page

def url2lines(url):
    page = get_page(url)

    if page is None:
        return []
    
    lines = html2lines(page)
    return lines

def line_correction(lines, max_size=100):
    out_lines = []
    for line in lines:
        if len(line) < 4:
            continue

        if len(line) > max_size:
            doc = nlp(line[:5000]) # We split lines into sentences, but for performance we take only the first 5k characters per line
            stack = ""
            for sent in doc.sents:
                if len(stack) > 0:
                    stack += " "
                stack += str(sent).strip()
                if len(stack) > max_size:
                    out_lines.append(stack)
                    stack = ""

            if len(stack) > 0:
                out_lines.append(stack)
        else:
            out_lines.append(line)
    
    return out_lines

def html2lines(page):
    out_lines = []

    if len(page.strip()) == 0 or page is None:
        return out_lines

    text = trafilatura.extract(page, config=DEFAULT_CONFIG)
    reset_caches()

    if text is None:
        return out_lines

    return text.split("\n") # We just spit out the entire page, so need to reformat later.