import pandas as pd
import random
import json
from pathlib import Path


data_sets = ["Chinese", "English", "IT", "Medicine", "Pokemon", "Science"]


def HyperMajorityVote(k, m):
    hyper_questions = {}
    majority_vote_all_answers = {}
    majority_vote = {}
    #Loading DataSet
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

        hyper_questions_set_size = k
        hyper_questions[data_set][hyper_questions_set_size] = {}

        seen_combinations = set()
        #Loading Permutations
        for _ in range(m): 
            shuffled_questions = list(range(questions_count))
            random.shuffle(shuffled_questions)

            for start in range(0, questions_count - k + 1, k):
                current_question_set = sorted(shuffled_questions[start:start + k])
               
                current_question_set_tuple = tuple(current_question_set)
                if current_question_set_tuple in seen_combinations:
                    continue
                answers_to_hyper_questions = df.iloc[current_question_set].T

                answers_set = answers_to_hyper_questions.value_counts().to_dict()
                seen_combinations.add(current_question_set_tuple)
                hyper_questions[data_set][hyper_questions_set_size][
                current_question_set_tuple
                ] = answers_set
    hyper_majority_vote = {}  
    questions_hyper_majority_vote = {}
    #Picking winning tuple for each hyper question
    for data_set, set_sizes in hyper_questions.items():
        for set_size, question_ranges in set_sizes.items():
            for question_range, answers_set in question_ranges.items():
                answer_tuple, count = next(iter(answers_set.items()))
                key=(data_set, question_range, answer_tuple)
                if  key not in hyper_majority_vote:
                     hyper_majority_vote[key] = 0
                hyper_majority_vote[key]+=1
 #Votes for each Question               
    for (data_set, question_range, answer_tuple), count in hyper_majority_vote.items():
        for question_number, answer in zip(question_range, answer_tuple):   
            key=(data_set, question_number, answer)
            if  key not in questions_hyper_majority_vote:
                questions_hyper_majority_vote[key] = 0
            questions_hyper_majority_vote[key]+=1
#Final hyperQuestions Majority vote
    final_hyper_mv = {}       
    for (data_set, question, answer), frequency in questions_hyper_majority_vote.items():
        key = (data_set, question)
        if key not in final_hyper_mv:
            final_hyper_mv[key] = (answer, frequency)
        else:
            _, current_freq = final_hyper_mv[key]
            if frequency > current_freq:
                final_hyper_mv[key] = (answer, frequency)

#Normal Majority Vote
    for (data_set, question, answer), frequency in majority_vote_all_answers.items(): 
        key = (data_set, question)
        if key not in majority_vote:
            majority_vote[key] = (answer, frequency)
        else:
            _, current_freq = majority_vote[key]
            if frequency > current_freq:
                majority_vote[key] = (answer, frequency)

    # print("Normal MV:")
    # print(majority_vote)

    # print("Hyper-MV:per Question ")
    # print(final_hyper_mv)
    return majority_vote, final_hyper_mv


if __name__ == "__main__":
    HyperMajorityVote(4, 30)