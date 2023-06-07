import os
import nltk
from nltk import word_tokenize
import numpy as np
from leven import levenshtein
from sklearn.cluster import DBSCAN, dbscan

def delete_if_exists(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

def pairwise_meteor(candidate, reference): # Todo this is not thread safe, no idea how to make it so
    return nltk.translate.meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(candidate))

def count_stats(candidate_dict, reference_dict):
    count_match = [0 for _ in candidate_dict]
    count_diff = [0 for _ in candidate_dict]

    for i, k in enumerate(candidate_dict.keys()):
      pred_parts = candidate_dict[k]
      tgt_parts = reference_dict[k]

      if len(pred_parts) == len(tgt_parts):
        count_match[i] = 1

      count_diff[i] = abs(len(pred_parts) - len(tgt_parts))

    count_match_score = np.mean(count_match)
    count_diff_score = np.mean(count_diff)

    return {
        "count_match_score": count_match_score,
        "count_diff_score": count_diff_score
    }

def f1_metric(candidate_dict, reference_dict, pairwise_metric):
    all_best_p = [0 for _ in candidate_dict]
    all_best_t = [0 for _ in candidate_dict]
    p_unnorm = []

    for i, k in enumerate(candidate_dict.keys()):
      pred_parts = candidate_dict[k]
      tgt_parts = reference_dict[k]

      best_p_score = [0 for _ in pred_parts]
      best_t_score = [0 for _ in tgt_parts]

      for p_idx in range(len(pred_parts)):
        for t_idx in range(len(tgt_parts)):
          #meteor_score = pairwise_meteor(pred_parts[p_idx], tgt_parts[t_idx])
          metric_score = pairwise_metric(pred_parts[p_idx], tgt_parts[t_idx])

          if metric_score > best_p_score[p_idx]:
            best_p_score[p_idx] = metric_score

          if metric_score > best_t_score[t_idx]:
            best_t_score[t_idx] = metric_score

      all_best_p[i] = np.mean(best_p_score) if len(best_p_score) > 0 else 1.0
      all_best_t[i] = np.mean(best_t_score) if len(best_t_score) > 0 else 1.0       

      p_unnorm.extend(best_p_score) 

    p_score = np.mean(all_best_p)
    r_score = np.mean(all_best_t)
    avg_score = (p_score + r_score) / 2
    f1_score = 2 * p_score * r_score / (p_score + r_score + 1e-8)

    p_unnorm_score = np.mean(p_unnorm)

    return {
        "p": p_score,
        "r": r_score,
        "avg": avg_score,
        "f1": f1_score,
        "p_unnorm": p_unnorm_score,
    }

def edit_distance_dbscan(data):
  # Inspired by https://scikit-learn.org/stable/faq.html#how-do-i-deal-with-string-data-or-trees-graphs
  def lev_metric(x, y):
    i, j = int(x[0]), int(y[0])
    return levenshtein(data[i], data[j])

  X = np.arange(len(data)).reshape(-1, 1)

  clustering = dbscan(X, metric=lev_metric, eps=20, min_samples=2, algorithm='brute')
  return clustering

def compute_all_pairwise_edit_distances(data):
  X = np.empty((len(data), len(data)))

  for i in range(len(data)):
    for j in range(len(data)):
      X[i][j] = levenshtein(data[i], data[j])

  return X

def compute_all_pairwise_scores(src_data, tgt_data, metric):
  X = np.empty((len(src_data), len(tgt_data)))

  for i in range(len(src_data)):
    for j in range(len(tgt_data)):
      X[i][j] = (metric(src_data[i], tgt_data[j]))

  return X

def compute_all_pairwise_meteor_scores(data):
  X = np.empty((len(data), len(data)))

  for i in range(len(data)):
    for j in range(len(data)):
      X[i][j] = (pairwise_meteor(data[i], data[j]) + pairwise_meteor(data[j], data[i])) / 2

  return X

def edit_distance_custom(data, X, eps=0.5, min_samples=3):
  clustering = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples).fit(X)
  return clustering.labels_
