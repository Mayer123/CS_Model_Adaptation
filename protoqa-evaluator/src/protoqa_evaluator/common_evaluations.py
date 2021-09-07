from functools import partial

import numpy as np
from protoqa_evaluator.evaluation import general_eval
from protoqa_evaluator.scoring import wordnet_score

__all__ = [
    "max_answers",
    "max_incorrect",
    "exact_match_all_eval_funcs",
    "wordnet_all_eval_funcs",
    "all_eval_funcs",
    "fast_money",
    "family_feud",
    "set_intersection",
    "hard_set_intersection",
]

max_answers = {
    f"Max Answers - {k}": partial(general_eval, max_pred_answers=k)
    for k in [None, 1, 3, 5, 10]
}
max_incorrect = {
    f"Max Incorrect - {k}": partial(general_eval, max_incorrect=k)
    for k in [None, 1, 3, 5]
}
exact_match_all_eval_funcs = {**max_answers, **max_incorrect}
wordnet_all_eval_funcs = {
    k: partial(v, score_func=wordnet_score, score_matrix_transformation=np.round)
    for k, v in exact_match_all_eval_funcs.items()
}
all_eval_funcs = {
    "exact_match": exact_match_all_eval_funcs,
    "wordnet": wordnet_all_eval_funcs,
}

fast_money = partial(general_eval, max_pred_answers=1)
family_feud = partial(general_eval, max_incorrect=3)
set_intersection = partial(general_eval, assign_cluster_scores=False)
hard_set_intersection = partial(set_intersection, score_matrix_transformation=np.round)
