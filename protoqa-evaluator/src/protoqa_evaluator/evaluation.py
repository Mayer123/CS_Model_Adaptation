from .data_processing import QuestionAndAnswerClusters, default_string_preprocessing
from .scoring import *
import statistics
import numpy as np
from typing import *


def multiple_evals(
    eval_func_dict: Dict[str, Callable],
    question_data: Dict[str, QuestionAndAnswerClusters],
    answers_dict: Dict[str, List[str]],
    optimal_ranking: bool = False,
) -> Dict[str, Dict[str, float]]:
    eval_details = {}
    for name, eval_func in eval_func_dict.items():
        print(f"Evaluating {name}...", flush=True)
        eval_details[name] = evaluate(
            evaluation_func=eval_func,
            question_data=question_data,
            answers_dict=answers_dict,
            optimal_ranking=optimal_ranking,
        )
        eval_score = statistics.mean(x.score for x in eval_details[name].values())
        print(f"{name}: {eval_score}")
    return eval_details


def evaluate(
    evaluation_func: Callable,
    question_data: Dict[str, QuestionAndAnswerClusters],
    answers_dict: Dict[str, List[str]],
    data_preprocessing: Optional[Callable] = None,
    optimal_ranking: bool = False,
) -> Dict[str, float]:
    scores = dict()
    for qid, pred_answers in answers_dict.items():
        true_q = question_data[qid]
        if data_preprocessing is not None:
            true_q, pred_answers = data_preprocessing(true_q, answers_dict)
        true_answers = true_q.answer_clusters.copy()
        scores[qid] = evaluation_func(
            pred_answers,
            true_answers,
            question_string=true_q.question,
            optimal_ranking=optimal_ranking,
        )
    return scores


class EvalResult(NamedTuple):
    score: float
    score_matrix: np.ndarray
    answer_assignment: dict

    def __eq__(self, other):
        return (
            self.score == other.score
            and (self.score_matrix == other.score_matrix).all()
            and self.answer_assignment == other.answer_assignment
        )


def general_eval(
    pred_answers,
    true_answers,
    *,
    max_pred_answers: Optional[int] = None,
    max_incorrect: Optional[int] = None,
    string_preprocessing: Callable = default_string_preprocessing,
    question_string: str = "question string",
    score_func: Callable = exact_match,
    cluster_score_func: Callable = cluster_score,
    cluster_reduction_func: Callable = np.max,
    score_matrix_transformation: Optional[Callable] = None,
    assign_cluster_scores: bool = True,
    calc_oracle_score: bool = True,
    optimal_ranking: bool = False,
) -> EvalResult:
    if max_pred_answers is not None and not optimal_ranking:
        pred_answers = pred_answers[:max_pred_answers]
    pred_answers = [string_preprocessing(pred_answer) for pred_answer in pred_answers]
    score_matrix = cluster_score_func(
        pred_answers,
        true_answers,
        question_string=question_string,
        score_func=score_func,
        cluster_reduction_func=cluster_reduction_func,
    )
    # score_matrix has values in [0,1] at this point
    if score_matrix_transformation is not None:
        score_matrix = score_matrix_transformation(score_matrix)
    if max_incorrect is not None and not optimal_ranking:
        score_matrix = limit_total_wrong(score_matrix, max_incorrect)
    if assign_cluster_scores:
        score_matrix *= np.array(list(true_answers.values()))[None]
    score, row_ind, col_ind = get_optimal_score(score_matrix)

    if optimal_ranking:
        reward_and_ind = [
            (score_matrix[row_ind[z], col_ind[z]], row_ind[z], col_ind[z])
            for z in range(len(row_ind))
        ]
        sorted_by_reward = sorted(reward_and_ind, key=lambda z: z[0], reverse=True)
        _, row_ind, col_ind = zip(*sorted_by_reward)
        row_ind = np.array(row_ind)
        col_ind = np.array(col_ind)
        if max_pred_answers is not None:
            row_ind = row_ind[:max_pred_answers]
            col_ind = col_ind[:max_pred_answers]
        if max_incorrect is not None:
            for i in range(len(row_ind)):
                if score_matrix[row_ind[i], col_ind[i]] == 0:
                    break
            row_ind = row_ind[:i]
            col_ind = col_ind[:i]
        score = score_matrix[row_ind, col_ind].sum()

    answer_assignment = dict()
    true_answers_list = list(true_answers.keys())
    for r, c in zip(row_ind, col_ind):
        # answer_assignment[pred_answers[r]] = (
        #     true_answers_list[c] if score_matrix[r, c] > 0 else None
        # )
        if score_matrix[r, c] > 0:
            answer_assignment[pred_answers[r]] = true_answers_list[c]
        else:
            if np.argmax(score_matrix[r]) > 0:
                answer_assignment[pred_answers[r]] = '#####'
            else:
                answer_assignment[pred_answers[r]] = None
    if calc_oracle_score:
        oracle_answers = sorted(
            list(true_answers.keys()), key=lambda z: true_answers[z], reverse=True
        )
        if isinstance(oracle_answers[0], frozenset):
            oracle_answers = [ans for (ans, *_) in oracle_answers]
        oracle_score, *_ = general_eval(
            pred_answers=oracle_answers,
            true_answers=true_answers,
            max_pred_answers=max_pred_answers,
            max_incorrect=max_incorrect,
            string_preprocessing=string_preprocessing,
            question_string=question_string,
            score_func=score_func,
            cluster_score_func=cluster_score_func,
            cluster_reduction_func=cluster_reduction_func,
            score_matrix_transformation=score_matrix_transformation,
            assign_cluster_scores=assign_cluster_scores,
            calc_oracle_score=False,
            optimal_ranking=False,
        )
        score /= oracle_score
    return EvalResult(
        score=score, score_matrix=score_matrix, answer_assignment=answer_assignment
    )


# WordNet Similarity


# Direct implementations of some of the simpler algorithms,
# without the functional structure of the general setting.
# Useful for testing, in case something in the more general setting goes wrong.
def naive_family_feud(
    pred_answers: List[str],
    true_answers: Dict[str, int],
    *args,
    max_incorrect: int = 3,
    **kwargs,
) -> float:
    pred_answers = pred_answers.copy()
    true_answers = true_answers.copy()
    score = 0
    max_score = sum(true_answers.values())
    incorrect = 0
    for i, answer in enumerate(pred_answers):
        try:
            score += true_answers.pop(answer)
        except KeyError:
            incorrect += 1
            if incorrect >= max_incorrect:
                break
    score /= max_score
    return score


def naive_fast_money(pred_answers, true_answers):
    pred_answers = pred_answers.copy()
    true_answers = true_answers.copy()
    score = true_answers.get(pred_answers[0], 0)
    score /= max(true_answers.values())
    return score
