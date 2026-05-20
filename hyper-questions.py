import pandas as pd
import random
import json
from pathlib import Path

data_sets = ["Chinese", "English", "IT", "Medicine", "Pokemon", "Science"]


def HyperMajorityVote(k, m):
    hyper_questions = {}
    majority_vote_all_answers = {}
    majority_vote = {}

    for data_set in data_sets:
        data_set_path = Path("datasets") / data_set / "answer.csv"
        hyper_questions[data_set] = {}
        df = pd.read_csv(data_set_path, header=None)
        questions_count = len(df)
        for question, row in df.iterrows():
            for _, choice in row.items():
                key = (data_set, question, choice)
                if key not in majority_vote_all_answers:
                    majority_vote_all_answers[key] = 0
                majority_vote_all_answers[key] += 1

        hyper_questions_set_size = 3
        while hyper_questions_set_size <= k:
            hyper_questions[data_set][hyper_questions_set_size] = {}

            sets_num = questions_count * m // hyper_questions_set_size
            seen_combinations = set()
            for _ in range(sets_num):
                current_question_set = sorted(random.sample(range(questions_count), k))
                current_question_set_tuple = tuple(current_question_set)
                if current_question_set_tuple in seen_combinations:
                    continue
                seen_combinations.add(current_question_set_tuple)
                answers_to_hyper_questions = df.iloc[current_question_set][:].T

                answers_set = answers_to_hyper_questions.value_counts().to_dict()
                hyper_questions[data_set][hyper_questions_set_size][
                    current_question_set_tuple
                ] = answers_set
            hyper_questions_set_size += 1
        print(hyper_questions)

    hyper_majority_vote = {}
    questions = {}
    for data_set, set_sizes in hyper_questions.items():
        for set_size, question_ranges in set_sizes.items():
            for question_range, answers_set in question_ranges.items():
                answer_tuple, count = next(iter(answers_set.items()))
                hyper_majority_vote[(data_set, question_range, answer_tuple)] = count

    for (data_set, question_range, answer_tuple), count in hyper_majority_vote.items():
        range_start = question_range.start
        for index, answer in enumerate(answer_tuple):
            question_number = range_start + index
            questions[(data_set, question_number, answer)] = count

    for (data_set, question, answer), frequency in majority_vote_all_answers.items():
        key = (data_set, question)
        if key not in majority_vote:
            majority_vote[key] = (answer, frequency)
        else:
            current_ans, current_freq = majority_vote[key]
            if frequency > current_freq:
                majority_vote[key] = (answer, frequency)

    print("Normal MV:")
    print(majority_vote)

    print("Hyper-MV: per K ", k)
    print(hyper_majority_vote)

    print("Hyper-MV:per Question ")
    print(questions)


HyperMajorityVote(4, 6)
