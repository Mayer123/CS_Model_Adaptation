import warnings
from functools import partial
from pathlib import Path

import pytest
from protoqa_evaluator.common_evaluations import (
    fast_money,
    family_feud,
    set_intersection,
    hard_set_intersection,
)
from protoqa_evaluator.data_processing import (
    load_question_answer_clusters_from_jsonl,
    save_to_jsonl,
    load_data_from_excel,
)
from protoqa_evaluator.evaluation import *

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from nltk.metrics.distance import jaro_winkler_similarity

try:
    import pandas
    import xlrd

    CROWDSOURCE_CONVERSION_TESTS = True
except ImportError:
    CROWDSOURCE_CONVERSION_TESTS = False

try:
    import torch
    from protoqa_evaluator.bert_scoring import (
        TransformerScoringModel,
        hard_bert_eval,
    )

    BERT_EVAL = True
except ImportError:
    BERT_EVAL = False


eval_methods = {
    "family_feud": family_feud,
    "fast_money": fast_money,
    "family_feud_2_incorrect": partial(family_feud, max_incorrect=2),
    "family_feud_5_incorrect": partial(family_feud, max_incorrect=5),
    "set_intersection": set_intersection,
    "soft_jaro_winkler_set_intersection": partial(
        set_intersection, score_func=jaro_winkler_similarity
    ),
    "hard_jaro_winkler_set_intersection": partial(
        hard_set_intersection,
        score_func=jaro_winkler_similarity,
    ),
    "hard_lcsubstring_set_int": partial(
        hard_set_intersection,
        score_func=longest_common_substring_score,
    ),
    "hard_lcsubseq_set_int": partial(
        hard_set_intersection,
        score_func=longest_common_subsequence_score,
    ),
    "hard_lcsubstring": partial(
        general_eval,
        score_func=longest_common_substring_score,
        score_matrix_transformation=np.round,
    ),
    "hard_lcsubseq": partial(
        general_eval,
        score_func=longest_common_subsequence_score,
        score_matrix_transformation=np.round,
    ),
    "fast_money_wn_sim": partial(
        fast_money,
        score_func=wordnet_score,
        score_matrix_transformation=np.round,
    ),
    "family_feud_wn_sim": partial(
        family_feud,
        score_func=wordnet_score,
        score_matrix_transformation=np.round,
    ),
}

answer_set_10_60_30 = {"10": 10, "60": 60, "30": 30}
answer_set_Apple10_Bananna60_Carrot30 = {"apple": 10, "bananna": 60, "carrot": 30}
answer_set_12_27_48 = {"12": 12, "27": 27, "48": 48}
answer_set_looks_smell = {"smell": 2, "face": 5}

test_data = (
    (
        "exact",
        ["60", "30", "10"],
        answer_set_10_60_30,
        {
            "family_feud": 1.0,
            "fast_money": 1.0,
            "set_intersection": 1.0,
            "soft_jaro_winkler_set_intersection": 1.0,
        },
    ),
    (
        "exact_less_than_100",
        ["48", "27", "12"],
        answer_set_12_27_48,
        {"family_feud": 1.0, "fast_money": 1.0, "set_intersection": 1.0},
    ),
    (
        "no_match",
        ["a", "b", "c", "d", "e"],
        answer_set_10_60_30,
        {"family_feud": 0.0, "fast_money": 0.0, "set_intersection": 0.0},
    ),
    (
        "scale_to_max",
        ["30", "a", "b"],
        answer_set_10_60_30,
        {"family_feud": 0.3, "fast_money": 0.5, "set_intersection": 1 / 3},
    ),
    (
        "wrong_order",
        ["10", "30", "60"],
        answer_set_10_60_30,
        {"family_feud": 1.0, "fast_money": 1 / 6, "set_intersection": 1.0},
    ),
    (
        "no_double_counting",
        ["60", "60", "60"],
        answer_set_10_60_30,
        {"family_feud": 0.6, "fast_money": 1, "set_intersection": 1 / 3},
    ),
    (
        "three_wrong",
        ["30", "X", "X", "10", "X", "60"],
        answer_set_10_60_30,
        {"family_feud": 0.4, "fast_money": 0.5, "set_intersection": 1.0},
    ),
    (
        "three_wrong_right_away",
        ["X", "X", "X", "10", "X", "60"],
        answer_set_10_60_30,
        {
            "family_feud": 0.0,
            "fast_money": 0.0,
            "family_feud_5_incorrect": 0.7,
            "set_intersection": 2 / 3,
        },
    ),
    (
        "two_wrong",
        ["30", "X", "10", "X", "60"],
        answer_set_10_60_30,
        {"family_feud": 1.0, "family_feud_2_incorrect": 0.4, "set_intersection": 1.0},
    ),
    (
        "many_repeats_should_not_penalize",
        ["30", "30", "30", "30", "30", "30", "60"],
        answer_set_10_60_30,
        {
            "family_feud": 0.9,
            "family_feud_2_incorrect": 0.9,
            "family_feud_5_incorrect": 0.9,
            "set_intersection": 2 / 3,
        },
    ),
    (
        "sloppy_input_answers",
        ["an Apple", "X", "the Banannnaa"],
        answer_set_Apple10_Bananna60_Carrot30,
        {
            "family_feud": 0.0,
            "set_intersection": 0.0,
            "hard_jaro_winkler_set_intersection": 2 / 3,
            "hard_lcsubstring_set_int": 1 / 3,
            "hard_lcsubseq_set_int": 2 / 3,
        },
    ),
    (
        "non_exact_match_answers",
        ["odor", "looks"],
        answer_set_looks_smell,
        {
            "family_feud": 0.0,
            "set_intersection": 0.0,
            "hard_lcsubstring_set_int": 0,
            "hard_lcsubseq_set_int": 0,
            "fast_money_wn_sim": 0.4,
            "family_feud_wn_sim": 1.0,
        },
    ),
)


def conv_to_param_dict(test_data):
    return {
        k + "][" + e[0]: (eval_methods[k], e[1], e[2], v)
        for e in test_data
        for k, v in e[3].items()
    }


test_data_param_dict = conv_to_param_dict(test_data)


@pytest.mark.parametrize(
    "eval_method, pred_answers, true_answers, expected",
    list(test_data_param_dict.values()),
    ids=list(test_data_param_dict.keys()),
)
def test_parametrized(eval_method, pred_answers, true_answers, expected):
    assert eval_method(pred_answers, true_answers)[0] == expected


@pytest.fixture()
def data_path():
    mod_path = Path(__file__).parent
    return mod_path / "data_stub.jsonl"


def test_load_data(data_path):
    load_question_answer_clusters_from_jsonl(data_path)


@pytest.fixture()
def question_data(data_path):
    return load_question_answer_clusters_from_jsonl(data_path)


def test_access_data(question_data):
    for i in range(5):
        question_data[f"q{i}"]


def test_evaluate_single_question(question_data):
    out = evaluate(
        family_feud, question_data, answers_dict={"q0": ["umbrella", "hat", "towel"]}
    )
    expected = {
        "q0": EvalResult(
            score=0.3838383838383838,
            score_matrix=np.array(
                [
                    [38.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            answer_assignment={
                "umbrella": frozenset(["umbrella"]),
                "hat": None,
                "towel": None,
            },
        )
    }
    assert out == expected


@pytest.fixture()
def answers_5():
    return {
        "q0": ["umbrella", "sunscreen", "towel", "sun glasses"],
        "q1": ["bed", "shower", "bathroom"],
        "q2": ["baby crying"],
        "q3": [
            "10"
        ],  # These aren't great... we might want to filter out answers which are purely numbers.
        "q4": ["40"],
    }


def test_evaluate_multiple_questions(answers_5, question_data):
    eval_output = evaluate(set_intersection, question_data, answers_dict=answers_5)
    out = {k: v.score for k, v in eval_output.items()}
    expected = {
        "q0": 2 / 6,
        "q1": 2 / 7,
        "q2": 0,
        "q3": 1 / 7,
        "q4": 1 / 5,
    }
    assert out == expected


def test_readme_example(question_data):
    soft_lcsubsequence_set_int = partial(
        general_eval,
        score_func=longest_common_subsequence_score,
        assign_cluster_scores=False,  # This is what makes it a set, it turns off the cluster counts
    )

    eval_output = evaluate(
        soft_lcsubsequence_set_int,
        question_data,
        answers_dict={"q0": ["umbrella", "hat", "sun glasses"]},
    )
    correct_output = {
        "q0": EvalResult(
            score=0.3896103896103896,
            score_matrix=np.array(
                [
                    [1.0, 0.33333333, 0.25, 0.3, 0.125, 0.125],
                    [0.125, 0.0, 0.42857143, 0.1, 0.0, 0.4],
                    [
                        0.27272727,
                        0.45454545,
                        0.45454545,
                        0.90909091,
                        0.09090909,
                        0.27272727,
                    ],
                ]
            ),
            answer_assignment={
                "umbrella": frozenset(["umbrella"]),
                "hat": frozenset(["sun hat"]),
                "sun glasses": frozenset(["sunglasses"]),
            },
        )
    }
    assert eval_output["q0"].score == correct_output["q0"].score
    assert eval_output["q0"].answer_assignment == correct_output["q0"].answer_assignment
    assert np.isclose(
        eval_output["q0"].score_matrix, correct_output["q0"].score_matrix
    ).all()


@pytest.fixture()
def crowdsource_excel_path():
    mod_path = Path(__file__).parent
    return mod_path / "crowdsource_data_stub.xlsx"


def test_load_crowdsource_data(crowdsource_excel_path):
    load_data_from_excel(crowdsource_excel_path)


@pytest.fixture()
def crowdsource_excel_data(crowdsource_excel_path):
    return load_data_from_excel(crowdsource_excel_path)


def test_num_questions_crowdsource_data(crowdsource_excel_data):
    assert len(crowdsource_excel_data) == 2


q_dict = {
    "r1q1": {
        "question": "Name something you’d find in outer space.",
        "normalized-question": "name something you’d find in outer space.",
        "raw-original-answers": {
            "moon": 1,
            "earth": 1,
            "comet": 1,
            "sun": 1,
            "clouds": 1,
            "meteor": 1,
            "asteroid": 1,
            "planet": 1,
            "star": 1,
        },
        "raw-answers-cleaned": {
            "moon": 1,
            "earth": 1,
            "comet": 1,
            "sun": 1,
            "clouds": 1,
            "meteor": 1,
            "asteroid": 1,
            "planet": 1,
            "star": 1,
        },
        # Not sure yet if we want to include this data or not.
        # 'annotator-data': [
        #     {
        #         'id': 1,
        #         'clusters': {
        #             'moon': {'moon'},
        #             'planet': {'earth', 'planet'},
        #             'meteor/asteroid/comet': {'meteor', 'asteroid', 'comet'},
        #             'star': {'sun', 'star'},
        #             'clouds': {'clouds'},
        #         },
        #     },
        #     {
        #         'id': 2,
        #         'clusters': {
        #             'space rock': {'moon', 'comet', 'meteor', 'asteroid'},
        #             'earth': {'earth'},
        #             'star': {'star', 'sun'},
        #             'clouds': {'?'},
        #             'planet': {'planet'},
        #         },
        #     },
        # ]
        "answers-cleaned": {
            frozenset(["moon"]): 1,
            frozenset(["earth", "planet"]): 2,
            frozenset(["sun", "star"]): 2,
            frozenset(["comet", "meteor", "asteroid"]): 3,
        },
        "source": "crowdsource_data_stub.xlsx",
        "questionid": "r1q1",
        "sourceid": "1",
    },
    "r1q2": {
        "question": "Name a type of produce.",
        "normalized-question": "name a type of produce.",
        "raw-original-answers": {
            "aaple": 1,
            "banana": 1,
            "broccoli": 2,
            "na": 1,
            "wheat": 1,
            "fruit": 1,
            "fruits": 1,
            "vegetable": 1,
            "fruit or vegetable": 1,
        },
        "raw-answers-cleaned": {
            "apple": 1,
            "banana": 1,
            "broccoli": 2,
            "na": 1,
            "wheat": 1,
            "fruit": 2,
            "fruits": 1,
            "vegetable": 1,
        },
        "answers-cleaned": {
            frozenset(["apple", "banana", "fruit", "fruits"]): 5,
            frozenset(["broccoli", "vegetable"]): 3,
        },
        "source": "crowdsource_data_stub.xlsx",
        "questionid": "r1q2",
        "sourceid": "2",
    },
}


@pytest.mark.parametrize(
    "qid, key, value",
    [(qid, key, value) for qid, q in q_dict.items() for key, value in q.items()],
    ids=[f"{qid}][{key}" for qid, q in q_dict.items() for key in q.keys()],
)
def test_crowdsource_data(crowdsource_excel_data, qid, key, value):
    assert crowdsource_excel_data[qid][key] == value


crowdsource_test_data = (
    (
        "exact_match",
        ["fruit", "vegetable"],
        q_dict["r1q2"]["answers-cleaned"],
        {
            "family_feud": 1.0,
            "fast_money": 1.0,
            "set_intersection": 1.0,
            "hard_lcsubstring_set_int": 1.0,
            "hard_lcsubseq_set_int": 1.0,
            "hard_lcsubstring": 1.0,
            "hard_lcsubseq": 1.0,
        },
    ),
    (
        "misspelling",
        ["aaple", "brocoli"],
        q_dict["r1q2"]["answers-cleaned"],
        {
            "family_feud": 0.0,
            "fast_money": 0.0,
            "set_intersection": 0.0,
            "hard_lcsubstring_set_int": 0.5,
            "hard_lcsubseq_set_int": 1.0,
            "hard_lcsubstring": 5 / 8,
            "hard_lcsubseq": 1.0,
        },
    ),
    (
        "single_element_category",
        ["moon"],
        q_dict["r1q1"]["answers-cleaned"],
        {"family_feud": 1 / 8, "fast_money": 1 / 3, "set_intersection": 1 / 4},
    ),
    (
        "no_category_label_points",
        ["space rock"],
        q_dict["r1q1"]["answers-cleaned"],
        {
            "family_feud": 0.0,
            "fast_money": 0.0,
            "set_intersection": 0.0,
            "hard_lcsubstring": 0.0,
            "hard_lcsubseq": 0.0,
        },
    ),
    (
        "multiple_possible_assignments",
        ["broccoli/banana", "an apple"],
        q_dict["r1q2"]["answers-cleaned"],
        {
            "family_feud": 0.0,
            "fast_money": 0.0,
            "set_intersection": 0.0,
            "hard_lcsubstring": 1.0,
            "hard_lcsubseq": 1.0,
        },
    ),
    (
        "no_double_points_for_surface_forms",
        ["apple", "banana", "fruit", "fruits"],
        q_dict["r1q2"]["answers-cleaned"],
        {
            "family_feud": 5 / 8,
            "fast_money": 1.0,
            "set_intersection": 1 / 2,
            "hard_lcsubstring": 5 / 8,
            "hard_lcsubseq": 5 / 8,
        },
    ),
    (
        "no_double_points_for_partial_matches",
        ["apple", "aapple", "appple", "apples"],
        q_dict["r1q2"]["answers-cleaned"],
        {
            "family_feud": 5 / 8,
            "fast_money": 1.0,
            "set_intersection": 1 / 2,
            "hard_lcsubstring": 5 / 8,
            "hard_lcsubseq": 5 / 8,
        },
    ),
    (
        "no_match",
        ["a", "b", "c", "d", "e"],
        q_dict["r1q1"]["answers-cleaned"],
        {
            "family_feud": 0,
            "fast_money": 0,
            "set_intersection": 0,
            "hard_lcsubstring": 0,
            "hard_lcsubseq": 0,
        },
    ),
)

crowdsource_param_dict = conv_to_param_dict(crowdsource_test_data)


@pytest.mark.parametrize(
    "eval_method, pred_answers, true_answers, expected",
    list(crowdsource_param_dict.values()),
    ids=list(crowdsource_param_dict.keys()),
)
def test_crowdsource_eval(eval_method, pred_answers, true_answers, expected):
    assert eval_method(pred_answers, true_answers)[0] == expected


crowdsource_answers = {
    "r1q1": ["star", "galaxy", "dark matter"],
    "r1q2": ["apple", "broccoli", "asparagus", "orange"],
}

crowdsource_eval_result = {
    "r1q1": EvalResult(
        score=0.25,
        score_matrix=np.array(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        answer_assignment={
            "star": frozenset({"star", "sun"}),
            "galaxy": None,
            "dark matter": None,
        },
    ),
    "r1q2": EvalResult(
        score=1.0,
        score_matrix=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
        answer_assignment={
            "apple": frozenset({"banana", "fruits", "apple", "fruit"}),
            "broccoli": frozenset({"broccoli", "vegetable"}),
        },
    ),
}

q_qa_clusters = {
    k: QuestionAndAnswerClusters(
        v["questionid"], v["normalized-question"], v["answers-cleaned"]
    )
    for k, v in q_dict.items()
}


def test_crowdsource_eval_mult():
    out = evaluate(set_intersection, q_qa_clusters, answers_dict=crowdsource_answers)
    expected = crowdsource_eval_result
    assert out == expected


@pytest.fixture()
def crowdsource_jsonl_path():
    mod_path = Path(__file__).parent
    return mod_path / "crowdsource_data_stub.jsonl"


@pytest.fixture()
def crowdsource_jsonl_data(crowdsource_jsonl_path):
    return load_question_answer_clusters_from_jsonl(crowdsource_jsonl_path)


def test_crowdsource_eval_on_jsonl(crowdsource_jsonl_data):
    assert (
        evaluate(
            set_intersection, crowdsource_jsonl_data, answers_dict=crowdsource_answers
        )
        == crowdsource_eval_result
    )
