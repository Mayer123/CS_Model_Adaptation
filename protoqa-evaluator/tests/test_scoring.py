from typing import *
import pytest

from protoqa_evaluator.scoring import *


class AnswerScoreExample(NamedTuple):
    pred_answer: str
    true_answer: str
    funcs_to_eval: Dict[Callable, float]


# shorthand
ASE = AnswerScoreExample

test_answer_score_examples = {
    "wn_same_synset": ASE(
        "jump",
        "startle",
        {
            wordnet_score: 1,
            exact_match: 0,
            longest_common_substring_score: 0,
            longest_common_subsequence_score: 0,
        },
    ),
    "wn_oov": ASE("oov", "oov", {wordnet_score: 1}),
    "wn_partial_oov": ASE("oov plant", "oov flora", {wordnet_score: 1}),
    "wn_should_not_match": ASE(
        "ear muffs", "ear wax", {wordnet_score: 0.5, wordnet_wup_score: 2 / 3}
    ),
    "wn_gum": ASE("gum", "chewing gum", {wordnet_score: 1}),
}


def convert_to_param_dict(test_data):
    return {
        func.__name__
        + "]["
        + test_name: (func, ase.pred_answer, ase.true_answer, expected)
        for test_name, ase in test_data.items()
        for func, expected in ase.funcs_to_eval.items()
    }


test_answer_score_examples_param = convert_to_param_dict(test_answer_score_examples)


@pytest.mark.parametrize(
    "func, pred_answer, true_answer, expected",
    list(test_answer_score_examples_param.values()),
    ids=list(test_answer_score_examples_param.keys()),
)
def test_answer_score_example(func, pred_answer, true_answer, expected):
    assert func(pred_answer, true_answer) == expected
