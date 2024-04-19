# Averitec

This repository maintains the dataset and baseline described in our paper [AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web](https://arxiv.org/abs/2305.13117). 

## Format

The dataset is formatted as JSON, with each split located as a separate file in the *data*-folder. Each claim is an object of the following form:

```
{
    "claim": "The claim text itself",
    "required_reannotation": "True or False. Denotes that the claim received a second round of QG-QA and quality control annotation.",
    "label": "The annotated verdict for the claim",
    "justification": "A textual justification explaining how the verdict was reached from the question-answer pairs.",
    "claim_date": "Our best estimate for the date the claim first appeared",
    "speaker": "The person or organization that made the claim, e.g. Barrack Obama, The Onion.",
    "original_claim_url": "If the claim first appeared on the internet, a url to the original location",
    "cached_original_claim_url": "Where possible, an archive.org link to the original claim url",
    "fact_checking_article": "The fact-checking article we extracted the claim from",
    "reporting_source": "The website or organization that first published the claim, e.g. Facebook, CNN.",
    "location_ISO_code": "The location most relevant for the claim. Highly useful for search.",
    "claim_types": [
            "The types of the claim",
    ],
    "fact_checking_strategies": [
        "The strategies employed in the fact-checking article",
    ],
    "questions": [
        {
            "question": "A fact-checking question for the claim",
            "answers": [
                {
                    "answer": "The answer to the question",
                    "answer_type": "Whether the answer was abstractive, extractive, boolean, or unanswerable",
                    "source_url": "The source url for the answer",
                    "cached_source_url": "An archive.org link for the source url"
                    "source_medium": "The medium the answer appeared in, e.g. web text, a pdf, or an image.",
                }
            ]
        },
}
```

## Evaluation

The official evaluation script can be found in *eval.py*. To run the script, please use:

```
python eval.py --predictions your_predictions.json --references data/dev.json
```

## Baseline

Our baseline is a pipeline consisting of multiple steps. The first step is coarse retrieval, using Google search. Many pages will be downloaded, so please ensure access to a folder (e.g. "your_dataset_store") with plenty of space. Before running the script, you will need access to the Google Search API. You can obtain an API key and a CSE ID [here](programmablesearchengine.google.com). Please add these to the appropriate variables in coarse_retrieval/averitec_search.py.

```
python coarse_retrieval/prompt_question_generation.py --target_file data/dev.json > data/dev.generated_questions.json
python coarse_retrieval/averitec_search.py --averitec_file data/dev.generated_questions.json --store_folder your_dataset_store > search_results.tsv
```

The next step is question generation for each retrieved document, followed by reranking. This can be done with:

```
python retrieval_reranking/decorate_with_questions.py --url_file search_results.tsv --store_folder your_dataset_store > data/dev.bm25_questions.json
python trained_model_reranker.py --averitec_file data/dev.bm25_questions.json > data/dev.reranked_questions.json
```

Then, predict verdicts:

```
python veracity_prediction/veracity_prediction.py --averitec_file data/dev.reranked_questions.json > data/dev.verdicts.json
```

Finally, predict justifications:

```
python justification_production/trained_model_justification_generation.py --averitec_file data/dev.verdicts.json > data/dev.verdicts_and_justifications.json
```

## Citation

If you used our dataset or code, please cite our paper as:


```
@inproceedings{
    schlichtkrull2023averitec,
    title={{AV}eri{T}e{C}: A Dataset for Real-world Claim Verification with Evidence from the Web},
    author={Michael Sejr Schlichtkrull and Zhijiang Guo and Andreas Vlachos},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/forum?id=fKzSz0oyaI}
}
```

## License
<p align="center">
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
</p>
