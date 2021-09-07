import click


@click.group()
def main():
    """Evaluation and data processing for ProtoQA common sense QA dataset"""
    pass


@main.group()
def convert():
    """Functions for loading, converting, and saving data files"""
    pass


@convert.command()
@click.argument("xlsx_files", type=click.Path(exists=True), nargs=-1)
@click.argument("--output_jsonl", type=click.Path(), nargs=1)
def clustered(xlsx_files, output_jsonl):
    """Convert clustering XLSX files to JSONL"""
    from .data_processing import load_data_from_excel, save_to_jsonl

    q = dict()
    for idx, xlsx_file in enumerate(xlsx_files):
        next_q = load_data_from_excel(xlsx_file, idx + 1)
        q.update(next_q)
    return save_to_jsonl(output_jsonl, q)


@convert.command()
@click.argument("xlsx_file", type=click.Path())
@click.argument("question_jsonl", type=click.Path())
@click.argument("output_jsonl", type=click.Path())
@click.option("--include_do_not_use/--exclude_do_not_use", default=False)
@click.option("--allow_incomplete/--no_incomplete", default=False)
def crowdsource_to_ranked_list(
    xlsx_file, question_jsonl, output_jsonl, include_do_not_use, allow_incomplete
):
    """Convert human *crowdsourced* answers XLSX files to JSONL predictions file"""
    from .data_processing import (
        load_question_answer_clusters_from_jsonl,
        load_crowdsourced_xlsx_to_predictions,
        save_predictions_to_jsonl,
    )

    questions = load_question_answer_clusters_from_jsonl(question_jsonl)
    questions_to_ids = {q.question: q.question_id for q in questions.values()}
    predictions_dict = load_crowdsourced_xlsx_to_predictions(
        xlsx_file, questions_to_ids
    )
    save_predictions_to_jsonl(predictions_dict, output_jsonl)


@convert.command()
@click.argument("xlsx_file", type=click.Path())
@click.argument("question_jsonl", type=click.Path())
@click.argument("output_jsonl", type=click.Path())
@click.option("--include_do_not_use/--exclude_do_not_use", default=False)
@click.option("--allow_incomplete/--no_incomplete", default=False)
def ranking(
    xlsx_file, question_jsonl, output_jsonl, include_do_not_use, allow_incomplete
):
    """Convert human *ranking* task XLSX files to JSONL"""
    from .data_processing import (
        load_ranking_data,
        load_jsonl_to_list,
        convert_ranking_data_to_answers,
        save_list_to_jsonl,
    )

    ranking_data = load_ranking_data(xlsx_file)
    question_list = load_jsonl_to_list(question_jsonl)
    answers_list = convert_ranking_data_to_answers(
        ranking_data, question_list, allow_incomplete
    )
    if not include_do_not_use:
        do_not_use = {q["questionid"] for q in question_list if q["do-not-use"]}
        answers_ids = {next(a.keys()) for a in answers_list}
        num_int = len(do_not_use.intersection(answers_ids))
        print(
            f"Removing {num_int} answers whose associated questions were marked as DO_NOT_USE"
        )
        answers_list = [a for a in answers_list if next(a.keys()) not in do_not_use]
    save_list_to_jsonl(answers_list, output_jsonl)


@convert.command()
@click.argument("input_jsonl", type=click.Path(exists=True))
@click.argument("output_jsonl", type=click.Path())
def old_jsonl_to_new(input_jsonl, output_jsonl):
    """Convert old format jsonl to new"""
    from .data_processing import (
        load_jsonl_to_list,
        convert_old_list_to_new,
        save_list_to_jsonl,
    )

    print(f"Loading {input_jsonl}...", flush=True, end=None)
    input_list = load_jsonl_to_list(input_jsonl)
    print("done!")
    converted_list = convert_old_list_to_new(input_list)
    print(f"Saving {output_jsonl}...", flush=True, end=None)
    save_list_to_jsonl(converted_list, output_jsonl)
    print("done!")


@main.command()
@click.argument("targets_jsonl", type=click.Path())
@click.argument("predictions_jsonl", type=click.Path())
@click.option(
    "--similarity_function",
    default="wordnet",
    type=click.Choice(["exact_match", "wordnet"], case_sensitive=False),
)
@click.option(
    "--scoring",
    default="answer_cluster_count",
    type=click.Choice(
        ["answer_cluster_count", "traditional_dcg"], case_sensitive=False
    ),
)
@click.option("--optimal_ranking", is_flag=True, default=False, help="")
def evaluate(
    targets_jsonl, predictions_jsonl, similarity_function, scoring, optimal_ranking
):
    """Run all evaluation metrics on model outputs"""
    from .data_processing import (
        load_question_answer_clusters_from_jsonl,
        load_predictions_from_jsonl,
    )
    from .evaluation import multiple_evals
    from protoqa_evaluator.common_evaluations import all_eval_funcs
    from functools import partial

    print(f"Using {similarity_function} similarity.", flush=True)
    targets = load_question_answer_clusters_from_jsonl(targets_jsonl)
    # In case questions in predictions_jsonl is a superset of those in targets_jsonl
    all_predictions = load_predictions_from_jsonl(predictions_jsonl)
    predictions = {k: all_predictions[k] for k in targets}
    eval_func_dict = all_eval_funcs[similarity_function]
    if scoring is "traditional_dcg":
        from .scoring import traditional_discounted_cumulative_gain

        eval_func_dict = {
            k: partial(
                v, score_matrix_transformation=traditional_discounted_cumulative_gain
            )
            for k, v in eval_func_dict.items()
        }

    multiple_evals(
        eval_func_dict=eval_func_dict,
        question_data=targets,
        answers_dict=predictions,
        optimal_ranking=optimal_ranking,
    )
