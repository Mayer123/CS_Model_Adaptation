## Evaluation Script for  ProtoQA question/answers.

Clone via git and install (preferably in the same virtual environment as your model) with pip:
```bash
conda activate protoqa-evaluator (or similar)
git clone <repo-url>
pip install -e protoqa-evaluator
```

This will install a command-line utility you can use for evaluation, eg.

```bash
protoqa_evaluator evaluate --similarity_function exact_match targets.jsonl predictions.jsonl
```

Note that the `predictions.jsonl` file should be a jsonl file where each line has a question id and a ranked list of answers, eg.

```
{"r1q1": ["age", "job", "name"]}
{"r1q2": ["fight", "sickness", "sleeping"]}
```

The `targets.jsonl` file should be in the evaluation format from https://github.com/iesl/protoqa-data.


There is also an API for programmatic evaluation. For example, to calculate a max incorrect @ 3 score with exact match similarity, you can run

```python
from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
from protoqa_evaluator.evaluation import general_eval, evaluate
from functools import partial

question_data = load_question_answer_clusters_from_jsonl('path/to/dataset_lines.jsonl')
max_incorrect_3 = partial(general_eval, max_predicted_answers=3)
evaluate(max_incorrect_3, question_data, answers_dict={'q0': ['umbrella', 'hat', 'sun glasses']})
# Returns {'q0': 0.3838383838}
```
As indicated above, model answers should be specified as a dict of (ranked) lists.

Common evaluation methods are available in [src/protoqa_evaluator/common_evaluations.py](src/protoqa_evaluator/common_evaluations.py). To run multiple evaluations, you can pass a dict of evaluation functions to `multiple_evals`, for example:

```python
from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
from protoqa_evaluator.evaluation import multiple_evals
from protoqa_evaluator.common_evaluations import exact_match_all_eval_funcs

question_data = load_question_answer_clusters_from_jsonl('path/to/dataset_lines.jsonl')
multiple_evals(exact_match_all_eval_funcs, question_data, answers_dict={'q0': ['umbrella', 'hat', 'sun glasses']})
```

### Creating a Custom Evaluation Method
It is easy to create your own evaluation method using the `general_eval`. For example, let's make a set intersection evaluation which simply tells us what percentage of the true answer clusters we got right, and let's also use `longest_common_subsequence_score`  as our answer scoring function so that 'sun glasses' gets counted in the 'sunglasses' cluster:
```python
from protoqa_evaluator.evaluation import general_eval
from protoqa_evaluator.scoring import longest_common_subsequence_score
from functools import partial

soft_lcsubsequence_set_int = partial(
    general_eval,
    score_func = longest_common_subsequence_score,
    assign_cluster_scores = False, # This is what makes it a set, it turns off the cluster counts
)

evaluate(soft_lcsubsequence_set_int, question_data,
         answers_dict={'q0': ['umbrella', 'hat', 'sun glasses']})
```
This will return a dict of `EvalResult` objects, as follows:
```python
{
'q0': EvalResult(
    score=0.3896103896103896,
    score_matrix=np.array([[1., 0.33333333, 0.25, 0.3, 0.125, 0.125],
                           [0.125, 0., 0.42857143, 0.1, 0., 0.4],
                           [0.27272727, 0.45454545, 0.45454545, 0.90909091, 0.09090909, 0.27272727]]),
    answer_assignment={'umbrella': 'umbrella', 'hat': 'sun hat', 'sun glasses': 'sunglasses'})
}
```

For each question, the score which is returned is the percentage out of the maximum which could have been received, ie. percentage of oracle score. (This is calculated automatically, regardless of evaluation method, by passing the actual answers back into the function.) In situations with partial scoring for answers, it is possible for a single answer to score positively with more than one cluster (eg. "sun hat" would get a positive score with "hat" and "sun glasses"). In these scenarios the evaluation always makes the optimal assignment of answers to clusters using the Munkres assignment algorithm.

### WordNet Evaluation
Setting the `score_func = wordnet_score` will allow evaluation using WordNet Synsets. By default this function will
1. Tokenize the answer and true strings, removing stop words.
2. Compare (contiguous groups of) tokens to see if they are in the same synset or are an exact string match.
3. Return a score based on the optimal matching of tokens.

You might prefer to use a different score between synsets, for example Wu-Palmer similarity (see [this StackExchange post](https://linguistics.stackexchange.com/questions/9084/what-do-wordnetsimilarity-scores-mean)). Due to the many processing steps, the actual WordNet synset score function is actually rather "deep in the stack", so overriding it requires overriding it at three levels. (In addition, the `wup_similarity` function can sometimes return `None`, so we need to wrap the function itself.)
```python
# This is needed to ensure the outputs are always floats.
def wup_similarity_wrapper(*args, **kwargs):
    sim = wn.wup_similarity(*args, **kwargs)
    if sim is None:
        sim = 0.0
    return sim

# We now need to override the default wordnet evaluation functions.
# There are three:

# 1. A function which compares synsets
wordnet_wup_synset_score = partial(wordnet_synsets_score, score_func=wup_similarity_wrapper)
# 2. A function which compares partitions (lists of contiguous tokens) of strings
wordnet_wup_partition_score = partial(wordnet_partition_score,
                                      score_func=lambda a, b: max(wordnet_wup_synset_score(a, b),
                                                                  exact_match(a, b)), # Fallback if not in WordNet
                                      )
# 3. A function which compares the surface-form strings
wordnet_wup_score = partial(wordnet_score, score_func=wordnet_wup_partition_score)
```
You can now pass `wordnet_wup_score` as the `score_func` to an evaluation method if you would like. (Note: there is no need for you to repeat the steps above, as it is already included in `family_feud_evaluator.scoring`. It is included here for demonstration purposes.)

### MLM Model Similarity
This uses the Hugging Face implementation of the transformer and SciKit Learn. You can install the prerequisites via
```bash
pip install -e protoqa-evaluator[mlm-similarity]
```

The MLM similarity method works by converting the answers to vector representations before passing them to the evaluation functions. We have to specify this preprocessing step as follows:
```python
from protoqa_evaluator.bert_scoring import TransformerScoringModel, hard_bert_eval 
bert_scoring_model = TransformerScoringModel() # this sets up the model and loads the weights
evaluate(hard_bert_eval, question_data, answers_dict={'q0': ['age','demeanor','social status']}, bert_scoring_model.preprocessing)
```

### Testing
The package has tests written with `pytest`. To install test dependencies you can run
```bash
pip install -e protoqa-evaluator[test]
```
You can run the tests with
```bash
pytest protoqa-evaluator
```
