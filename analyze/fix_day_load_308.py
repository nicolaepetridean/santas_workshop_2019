import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import analyze.load_santa_data as santa


data = pd.read_csv('D:\\jde\\projects\\santas_workshop_2019\\santadata\\family_data.csv', index_col='family_id')

cols = [f'choice_{i}' for i in range(10)]
choice_dict = data[cols].to_dict()

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

# from 100 to 1
days = list(range(N_DAYS,0,-1))
family_size_dict = data[['n_people']].to_dict()['n_people']


def return_family_data():
    data_load = santa.SantaDataLoad()
    df = data_load.load_family_initial_data("D:\\jde\\projects\\santas_workshop_2019\\santadata\\")
    return df


def load_solution_data(solution_file = "sample_submission_output_fix.csv"):
    data_load = santa.SantaDataLoad()
    df = data_load.load_solution_file("D:\\jde\\projects\\santas_workshop_2019\\santadata\\" + solution_file)
    return df


def return_family_sizes(df):
    return list(df['n_people'])


def get_cost_by_choice(family_size):
    """
    Input : num_members
    Output : [(choice number indices),(corresponding cost)]
    """

    cost_by_choice = {}

    cost_by_choice[1] = 50
    cost_by_choice[2] = 50 + 9 * family_size
    cost_by_choice[3] = 100 + 9 * family_size
    cost_by_choice[4] = 200 + 9 * family_size
    cost_by_choice[5] = 200 + 18 * family_size
    cost_by_choice[6] = 300 + 18 * family_size
    cost_by_choice[7] = 400 + 36 * family_size
    cost_by_choice[8] = 500 + (36 + 199) * family_size
    cost_by_choice[9] = 500 + (36 + 398) * family_size

    return list(zip(*cost_by_choice.items()))


def compute_daily_load(solution, initial_data):
    days_load = np.zeros(101)
    row = 0
    while row < solution.shape[0]:
        day = solution.iloc[row, 0]
        n_people = initial_data.iloc[row, 11]
        days_load[int(day)] += n_people
        row += 1
    print(" sum of all people is :" + str(np.sum(days_load)))
    print(" min of all days is :" + str(np.min(days_load)))
    print(" max of all days is :" + str(np.max(days_load)))

    return days_load


def compute_family_in_a_specific_day(solution, wanted_day):
    families = []
    row = 0
    while row < solution.shape[0]:
        day = solution.iloc[row, 0]
        if day == wanted_day:
            families.append(row)
        row += 1

    return families


def calculate_choice_id_and_days_per_family(solution, initial_data):
    family_choice_ids = np.zeros(5000)
    family_choice_days = {}
    row = 0
    while row < initial_data.shape[0]:
        family_id = initial_data.iloc[row, 0]
        day = solution.iloc[row, 0]

        choice = 0
        choices = np.zeros(10)
        for i in range(0, 10):
            if initial_data.iloc[row, i+1] == day:
                choice = i
            choices[i] = initial_data.iloc[row, i+1]
            family_choice_days[family_id] = choices

        family_choice_ids[int(family_id)] = choice
        row += 1

    return family_choice_ids, family_choice_days


if __name__ == "__main__":
    initial_data = return_family_data()
    family_size = return_family_sizes(data)
    solution = load_solution_data('new\\try_mixed_.csv')
    daily_load = compute_daily_load(solution, initial_data)
    choice_ids, choice_days = calculate_choice_id_and_days_per_family(solution, initial_data)

    for day_id, load in enumerate(daily_load):
        if load > 300:
            candidates = compute_family_in_a_specific_day(solution, day_id)
            fixed = False
            for fam_id in candidates:
                if family_size[fam_id] >= 5:
                    for day in choice_days[fam_id]:
                        if daily_load[int(day)] + family_size[fam_id] < 300:
                            solution['assigned_day'][fam_id] = day
                            fixed = True
                            break
                if fixed:
                    break

    sub = pd.DataFrame(range(5000), columns=['family_id'])
    sub['assigned_day'] = solution['assigned_day']
    sub.to_csv('D:\\jde\\projects\\santas_workshop_2019\\santadata\\new\\try_mixed_.csv', index=False)


