import pandas as pd

from pathlib import Path

data_sets = ['Chinese', 'English', 'IT', 'Medicine', 'Pokemon', 'Science']
hyper_questions = {}
for data_set in data_sets:
    data_set_path = Path("datasets") / data_set /'answer.csv'
    hyper_questions[data_set] = {}
    df = pd.read_csv(data_set_path, header= None)
    hyper_questions_set_size = 3
    print(data_set)
    while hyper_questions_set_size < 7:
        hyper_questions[data_set][hyper_questions_set_size] = {} 

        print(hyper_questions_set_size)
        current_row = 0
        sets_num = len(df) // hyper_questions_set_size + (1 if  len(df) % hyper_questions_set_size != 0 else 0)
        for i in range(sets_num):
            current_question_range = range(current_row, min(current_row+ hyper_questions_set_size, len(df)))
            answers_to_hyper_questions = df.iloc[current_question_range][:].T
            answers_set = answers_to_hyper_questions.value_counts().to_dict()
            hyper_questions[data_set][hyper_questions_set_size][current_question_range] = answers_set
            current_row += hyper_questions_set_size
        hyper_questions_set_size += 1

print(hyper_questions)
            
