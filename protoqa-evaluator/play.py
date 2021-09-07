from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl, load_predictions_from_jsonl, QuestionAndAnswerClusters, default_string_preprocessing
from functools import partial
import numpy as np
from typing import *
from itertools import product
from scipy.optimize import linear_sum_assignment
import statistics

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

def exact_match(pred_answer: str, true_answer: str) -> float:
    return float(pred_answer == true_answer)

def get_optimal_score(score_matrix: np.ndarray) -> Tuple[float, List[int], List[int]]:
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return score_matrix[row_ind, col_ind].sum(), row_ind, col_ind

def all_pairs_scores(
    a: Union[str, Iterable],
    b: Union[str, Iterable],
    score_func: Callable,
    reduction_func: Callable = lambda z: z,
    preprocess_func: Callable = lambda z: [z] if isinstance(z, str) else z,
) -> Union[np.ndarray, float]:
    """
    Generic function for pairwise comparisons. Takes strings or iterables a and b and
    a score function and returns the matrix of all pairwise scores between a and b.
    :param a: Typically a string or iterable of strings to compare with b.
    :param b: Typically a string or iterable of strings to compare with a.
    :param score_func: Function which accepts two arguments (a,b) and returns their score in [0,1]
    :param reduction_func: Function which accepts a matrix and (typically) returns a scalar
    :param preprocess_func: Function which is run on both a and b prior to anything else
    :param kwargs: passed on to the score_func
    :return: Matrix of pairwise scores or output of reduction function on this matrix
    """
    a, b = preprocess_func(a), preprocess_func(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    #print ('a', a)
    #print ('b', b)
    score_matrix = np.zeros((len(a), len(b)))
    for (a_idx, a_val), (b_idx, b_val) in product(enumerate(a), enumerate(b)):
        score_val = score_func(a_val, b_val)
        score_matrix[a_idx, b_idx] = score_val
        if not (0 <= score_val <= 1):
            warnings.warn(
                f"Score function did not return a value in [0,1]: "
                f"score_func({a_val}, {b_val}) = {score_val} with type {type(score_val)}"
            )
        #print ('a_val', 'b_val', a_val, b_val)
    #print (score_matrix)
    #exit(0)
    return reduction_func(score_matrix)

def cluster_score(
    pred_answers: List[str],
    true_answers: Union[Dict[str, int], Dict[frozenset, int]],
    question_string: str,
    score_func: Callable = exact_match,
    cluster_reduction_func: Callable = np.max,
) -> np.ndarray:
    true_ans, *_ = true_answers
    if isinstance(true_ans, frozenset):
        score_func = partial(
            all_pairs_scores,
            score_func=score_func,
            reduction_func=cluster_reduction_func,
        )
    return all_pairs_scores(pred_answers, true_answers, score_func)

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
        #print (true_q)
        true_answers = true_q.answer_clusters.copy()
        #print (pred_answers)
        scores[qid] = evaluation_func(
            pred_answers,
            true_answers,
            question_string=true_q.question,
            optimal_ranking=optimal_ranking,
        )
        #if scores[qid].score == 0:
        #    print (qid, scores[qid])
        #print (scores[qid])
        #exit(0)
    return scores

def limit_total_wrong(score_matrix: np.ndarray, k: int) -> np.ndarray:
    #print (score_matrix)
    answer_scores = score_matrix.max(axis=1)
    #print (answer_scores)
    #exit(0)
    incorrect = 0
    for i, a in enumerate(answer_scores):
        if a == 0:
            incorrect += 1
            if incorrect >= k:
                return score_matrix[: i + 1]
    return score_matrix

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
    #print ('pred_answers', pred_answers)
    #print ('true_answers', true_answers)
    score_matrix = cluster_score_func(
        pred_answers,
        true_answers,
        question_string=question_string,
        score_func=score_func,
        cluster_reduction_func=cluster_reduction_func,
    )
    #print (score_matrix)
    #exit(0)
    # score_matrix has values in [0,1] at this point
    if score_matrix_transformation is not None:
        score_matrix = score_matrix_transformation(score_matrix)
    if max_incorrect is not None and not optimal_ranking:
        score_matrix = limit_total_wrong(score_matrix, max_incorrect)
    #print (score_matrix)
    if assign_cluster_scores:
        score_matrix *= np.array(list(true_answers.values()))[None]
    #print (score_matrix)
    score, row_ind, col_ind = get_optimal_score(score_matrix)
    #print (score_matrix)
    #print (row_ind)
    #print (col_ind)
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
        if score_matrix[r, c] > 0:
            answer_assignment[pred_answers[r]] = true_answers_list[c]
        else:
            if np.argmax(score_matrix[r]) > 0:
                answer_assignment[pred_answers[r]] = '#####'
            else:
                answer_assignment[pred_answers[r]] = None
        #answer_assignment[pred_answers[r]] = (true_answers_list[c] if score_matrix[r, c] > 0 else None)
    #print (answer_assignment)
    #exit(0)
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

def find_salient(cluster):
    best = None
    for word in cluster:
        if best == None:   
            best = word
        else:
            if len(word) < len(best):
                best = word
    return best

if __name__ == '__main__':
    question_data = load_question_answer_clusters_from_jsonl('../data/dev/dev.crowdsourced.jsonl')
    predictions = load_predictions_from_jsonl('../../../RAG_workspace/bart_fine_nofix1/checkpoint-5500/mergeranked_list.jsonl')
    print (len(question_data), len(predictions))
    max_incorrect_3 = partial(general_eval, max_incorrect=3)
    this_q = {'r1q2':question_data['r1q2']}
    this_pred = {'r1q2': predictions['r1q2']}
    scores = evaluate(max_incorrect_3, question_data, answers_dict=predictions)
    print (np.mean([x.score for x in scores.values()]))
    for qid, preds in predictions.items():
        #print (question_data[qid].answer_clusters)
        #print (scores[qid])
        #print (predictions[qid])
        matched_answers = []
        overlap_answers = []
        wrong_answers = []
        missing_answers = []
        for o in predictions[qid]:
            if o in scores[qid].answer_assignment:
                if scores[qid].answer_assignment[o] is not None:
                    matched_answers.append(o)
                else:
                    overlap_answers.append(o)
            else:
                wrong_answers.append(o)
        for k, v in question_data[qid].answer_clusters.items():
            if k not in scores[qid].answer_assignment.values():
                missing_answers.append([find_salient(k), v])
        print ('matched_answers', matched_answers)
        print ('overlap answers', overlap_answers)
        print ('wrong answers', wrong_answers)
        print ('missing answers', missing_answers)
        #if qid == 'r1q2':
        #    print (scores[qid].answer_assignment)
        #    exit(0)
        #exit(0)