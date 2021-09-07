from protoqa_evaluator.data_processing import load_data_from_jsonl, load_predictions_from_jsonl
from protoqa_evaluator.evaluation import evaluate, max_answers, max_incorrect, exact_match_all_eval_funcs

if __name__ == '__main__':
    question_data = load_data_from_jsonl('../data/dev/crowdsource_dev.jsonl')
    prev_predictions = load_predictions_from_jsonl('../ProtoQA_GPT2/gpt2/gpt2ranked_list.jsonl')
    predictions = load_predictions_from_jsonl('../ProtoQA_GPT2/pretrainedranked_list.jsonl')
    print (len(question_data), len(predictions))
    for qid, pred_answers in predictions.items():
        true_q = question_data[qid]
        true_answers = true_q["answers-cleaned"].copy()
        print (true_q['normalized-question'])
        print (true_answers)
        print (pred_answers)
        print (prev_predictions[qid])
        for name, eval_func in exact_match_all_eval_funcs.items():
            score = eval_func(
            pred_answers,
            true_answers,
            question_string=true_q["normalized-question"],
            )
            score1 = eval_func(
            prev_predictions[qid],
            true_answers,
            question_string=true_q["normalized-question"],
            )
            print (name, score1.score, score.score)
        print ()