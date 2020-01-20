import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import analyze.load_santa_data as santa


data = pd.read_csv('/Users/nicolaepetridean/jde/projects/santas_workshop_2019/santadata/family_data.csv', index_col='family_id')

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
    df = data_load.load_family_initial_data("/Users/nicolaepetridean/jde/projects/santas_workshop_2019/santadata/")
    return df


def load_solution_data(solution_file = "sample_submission_output_fix.csv"):
    data_load = santa.SantaDataLoad()
    df = data_load.load_solution_file("/Users/nicolaepetridean/jde/projects/santas_workshop_2019/santadata/" + solution_file)
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

def calculate_choice_id_per_family(solution, initial_data):
    family_choice_ids = np.zeros(5000)
    row = 0
    while row < initial_data.shape[0]:
        family_id = initial_data.iloc[row, 0]
        day = solution.iloc[row, 0]

        choice = 0
        for i in range(1, 10):
            if initial_data.iloc[row, i] == day:
                choice = i - 1
                break

        family_choice_ids[int(family_id)] = choice
        row += 1

    return family_choice_ids


def get_choice_cost(solution, initial_data):
    days_cost = np.zeros(101)
    row = 0
    while row < solution.shape[0]:
        family_size = initial_data.iloc[row, 11]
        try:
            day = solution.iloc[row, 0]
        except:
            day = solution[row]

        choice = 9
        for i in range(1, 10):
            if initial_data.iloc[row, i] == day:
                choice = i - 1
        if choice > 0:
            days_cost[int(day)] += get_cost_by_choice(family_size)[1][choice-1]
        row += 1

    return days_cost


def get_choice_cost_per_family(solution, initial_data):
    days_cost = np.zeros(101)
    choice_cost = []
    row = 0
    while row < solution.shape[0]:
        family_size = initial_data.iloc[row, 11]
        try:
            day = solution.iloc[row, 0]
        except:
            day = solution[row]

        choice = 9
        for i in range(1, 10):
            if initial_data.iloc[row, i] == day:
                choice = i - 1
        if choice > 0:
            family_cost = get_cost_by_choice(family_size)[1][choice-1]
            days_cost[int(day)] += family_cost
            choice_cost.append((row, choice, family_cost, family_size))
        row += 1

    return days_cost, choice_cost

def get_sum_of_choice_cost_per_size(solution, initial_data):
    days_cost = np.zeros(101)
    choice_cost = []
    choice_cost_per_fam_size = np.zeros(11)
    row = 0
    while row < solution.shape[0]:
        family_size = initial_data.iloc[row, 11]
        try:
            day = solution.iloc[row, 0]
        except:
            day = solution[row]

        choice = 9
        for i in range(1, 10):
            if initial_data.iloc[row, i] == day:
                choice = i - 1
        if choice > 0:
            family_cost = get_cost_by_choice(family_size)[1][choice-1]
            days_cost[int(day)] += family_cost
            choice_cost.append((row, choice, family_cost, family_size))
            choice_cost_per_fam_size[family_size] += family_cost
        row += 1

    #assert np.sum(choice_cost_per_fam_size) == 66056
    return days_cost, choice_cost, choice_cost_per_fam_size


def get_choice_distribution_vs_family_size(solution, initial_data):
    row = 0
    ch_vs_fami_size = {}
    while row < solution.shape[0]:
        family_size = initial_data.iloc[row, 11]
        try:
            day = solution.iloc[row, 0]
        except:
            day = solution[row]

        choice = 9
        for i in range(1, 10):
            if initial_data.iloc[row, i] == day:
                choice = i - 1
        if family_size in ch_vs_fami_size.keys():
            choices_distrib = ch_vs_fami_size[family_size]
            if choice in choices_distrib.keys():
                choices_distrib[choice] = choices_distrib[choice] + 1
            else:
                choices_distrib[choice] = 1
            ch_vs_fami_size[family_size] = choices_distrib
        else:
            choices_distrib = {}
            choices_distrib[choice] = 1
            ch_vs_fami_size[family_size] = choices_distrib

        row += 1

    return ch_vs_fami_size

if __name__ == "__main__":
    initial_data = return_family_data()

    solution = load_solution_data('sample_submission_69202_new.csv')

    daily_load = compute_daily_load(solution, initial_data)

    days_cost, ch_cost = get_choice_cost_per_family(solution, initial_data)

    days_cost, ch_cost, per_fam_size = get_sum_of_choice_cost_per_size(solution, initial_data)

    per_fam_size = get_choice_distribution_vs_family_size(solution, initial_data)

    choice_cost = np.sum(get_choice_cost(solution, initial_data))

    choices = calculate_choice_id_per_family(solution, initial_data)

    print('statistics per choice ' + str(np.unique(choices, return_counts=True)))
    print('choice_cost :' + str(choice_cost))

