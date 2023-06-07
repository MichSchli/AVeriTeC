
from utils import pairwise_meteor, compute_all_pairwise_scores
import numpy as np
import argparse
import json
import scipy
import sklearn

parser = argparse.ArgumentParser(description='Evaluation script for averitec.')
parser.add_argument('--predictions', default="your_prediction_here.json", help='')
parser.add_argument('--references', default="data/dev.json", help='')
args = parser.parse_args()

class AveritecEvaluator:

    verdicts = [
        "Supported", 
        "Refuted", 
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking"
    ]
    pairwise_metric = None
    max_questions = 10
    metric = None
    averitec_reporting_levels = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]

    def __init__(self, metric="meteor"):
        self.metric = metric
        if metric == "meteor":
            self.pairwise_metric = pairwise_meteor

    def evaluate_justifications(self, srcs, tgts):
        scores = []
        for src, tgt in zip(srcs, tgts):
            # If there is no justification, fallback to qa-pairs (or string evidence)
            if "justification" not in src:
                src_strings = self.extract_full_comparison_strings(src)[:self.max_questions]
                pred_justifications = " ".join(src_strings)
            else:
                pred_justifications = src["justification"]
            score = self.pairwise_metric(pred_justifications, tgt["justification"])
            scores.append(score)
        return np.mean(scores)

    def evaluate_averitec_veracity_by_type(self, srcs, tgts, threshold=0.3):
        types = {}
        for src, tgt in zip(srcs, tgts):
            score = self.compute_pairwise_evidence_score(src, tgt)

            if score <= threshold:
                score = 0

            for t in tgt["claim_types"]:
                if t not in types:
                    types[t] = []

                types[t].append(score)

        return {t:np.mean(v) for t,v in types.items()}

    def evaluate_averitec_score(self, srcs, tgts):
        scores = []
        justification_scores = []
        for src, tgt in zip(srcs, tgts):
            score = self.compute_pairwise_evidence_score(src, tgt)

            if "justification" not in src:
                src_strings = self.extract_full_comparison_strings(src)[:self.max_questions]
                pred_justifications = " ".join(src_strings)
            else:
                pred_justifications = src["justification"]

            justification_score = self.pairwise_metric(pred_justifications, tgt["justification"])

            this_example_scores = [0.0 for _ in self.averitec_reporting_levels]
            this_example_j_scores = [0.0 for _ in self.averitec_reporting_levels]
            for i, level in enumerate(self.averitec_reporting_levels):
                if score > level:
                    this_example_scores[i] = src["label"] == tgt["label"]
                    this_example_j_scores[i] = justification_score

            scores.append(this_example_scores)
            justification_scores.append(this_example_j_scores)

        return np.mean(np.array(scores), axis=0), np.mean(np.array(justification_scores), axis=0)

    def evaluate_veracity(self, src, tgt):
        src_labels = [x["label"] for x in src]
        tgt_labels = [x["label"] for x in tgt]

        acc = np.mean([s == t for s,t in zip(src_labels, tgt_labels)])

        f1 = {self.verdicts[i]: x for i,x in enumerate(sklearn.metrics.f1_score(tgt_labels, src_labels, labels=self.verdicts, average=None))}
        f1["macro"] = sklearn.metrics.f1_score(tgt_labels, src_labels, labels=self.verdicts, average='macro')
        f1["acc"] = acc
        return f1

    def evaluate_questions_only(self, srcs, tgts):
        all_utils = []
        for src, tgt in zip(srcs, tgts):
            if "questions" not in src:
                # If there was no question, use the string evidence
                src_questions = self.extract_full_comparison_strings(src)[:self.max_questions]
            else:
                src_questions = [qa["question"] for qa in src["questions"][:self.max_questions]]
            tgt_questions = [qa["question"] for qa in tgt["questions"]]

            pairwise_scores = compute_all_pairwise_scores(src_questions, tgt_questions, self.pairwise_metric)

            assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)

            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
            reweight_term = 1 / float(len(tgt_questions))
            assignment_utility *= reweight_term

            all_utils.append(assignment_utility)

        return np.mean(all_utils)

    def get_n_best_qas(self, srcs, tgts, n=3):
        all_utils = []
        for src, tgt in zip(srcs, tgts):
            assignment_utility = self.compute_pairwise_evidence_score(src, tgt)

            all_utils.append(assignment_utility)

        idxs = np.argsort(all_utils)[::-1][:n]

        examples = [(srcs[i]["questions"] if "questions" in srcs[i] else srcs[i]["string_evidence"], tgts[i]["questions"], all_utils[i]) for i in idxs]

        return examples

    def compute_pairwise_evidence_score(self, src, tgt):
        src_strings = self.extract_full_comparison_strings(src)[:self.max_questions]
        tgt_strings = self.extract_full_comparison_strings(tgt)
        pairwise_scores = compute_all_pairwise_scores(src_strings, tgt_strings, self.pairwise_metric)
        assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)
        assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
        reweight_term = 1 / float(len(tgt_strings))
        assignment_utility *= reweight_term
        return assignment_utility

    def evaluate_questions_and_answers(self, srcs, tgts):
        all_utils = []
        for src, tgt in zip(srcs, tgts):
            src_strings = self.extract_full_comparison_strings(src)[:self.max_questions]
            tgt_strings = self.extract_full_comparison_strings(tgt)

            pairwise_scores = compute_all_pairwise_scores(src_strings, tgt_strings, self.pairwise_metric)

            assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)

            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
            reweight_term = 1 / float(len(tgt_strings))
            assignment_utility *= reweight_term

            all_utils.append(assignment_utility)

        return np.mean(all_utils)

    def extract_full_comparison_strings(self, example):
        example_strings = []
        if "questions" in example:
            for question in example["questions"]:
                # If the answers is not a list, make them a list:
                if not isinstance(question["answers"], list):
                    question["answers"] = [question["answers"]]
                    
                for answer in question["answers"]:
                    example_strings.append(question["question"] + " " + answer["answer"])
                    if "answer_type" in answer and answer["answer_type"] == "Boolean":
                        example_strings[-1] += ". " + answer["boolean_explanation"]

                if len(question["answers"]) == 0:
                    example_strings.append(question["question"] + " No answer could be found.")
        
        if "string_evidence" in example:
            for full_string_evidence in example["string_evidence"]:
                example_strings.append(full_string_evidence)

        return example_strings

with open(args.predictions) as f:
    j = json.load(f)
    predictions = j

with open(args.references) as f:
    j = json.load(f)
    references = j

def print_with_space(left, right, left_space = 50):
    print_spaces = " " * (40 - len(left))
    print(left + print_spaces + right)

print("AVeriTeC evaluation:")
print("====================")

scorer = AveritecEvaluator()
q_score = scorer.evaluate_questions_only(predictions, references)
print_with_space("Question-only score (HU-" +scorer.metric + "):", str(q_score))
p_score = scorer.evaluate_questions_and_answers(predictions, references)
print_with_space("Question-answer score (HU-" +scorer.metric + "):", str(p_score))
print("====================")
v_score = scorer.evaluate_veracity(predictions, references)
print("Veracity F1 scores:")
for k,v in v_score.items():
    print_with_space(" * "+k+":", str(v))
j_score = scorer.evaluate_justifications(predictions, references)
print_with_space("Justification score (" +scorer.metric + "):", str(j_score))
print("--------------------")
print("Averitec scores:")
v_score, j_score = scorer.evaluate_averitec_score(predictions, references)
for i, level in enumerate(scorer.averitec_reporting_levels):
    print_with_space(" * Veracity scores (" +scorer.metric + " @ " + str(level) + "):", str(v_score[i]))
    print_with_space(" * Justification scores (" +scorer.metric  + " @ " + str(level) + "):", str(j_score[i]))
print("--------------------")
type_scores = scorer.evaluate_averitec_veracity_by_type(predictions, references, threshold=0.2)
for t,v in type_scores.items():
    print_with_space(" * Veracity scores (" +t + "):", str(v))
print("--------------------")
type_scores = scorer.evaluate_averitec_veracity_by_type(predictions, references,  threshold=0.3)
for t,v in type_scores.items():
    print_with_space(" * Veracity scores (" +t + "):", str(v))
print("====================")
print("Printing 5 best examples in terms of QA pairs:")
best_examples = scorer.get_n_best_qas(predictions, references, n=5)
print(json.dumps(best_examples, indent=4))