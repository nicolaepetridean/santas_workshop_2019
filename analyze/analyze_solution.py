import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import analyze.load_santa_data as santa
import os


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

def load_occupancy_data(solution_file = "occ_2_.csv"):
    occ = pd.read_csv("/Users/nicolaepetridean/jde/projects/santas_workshop_2019/santadata/" + solution_file, header=None, usecols=[0,1])[1]
    return np.concatenate(([0], np.array(occ)))

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


def compute_daily_load_2(solution, initial_data):
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
    families_cost = np.zeros(101)
    family = 0
    while family < solution.shape[0]:
        family_size = initial_data.iloc[family, 11]
        try:
            day = solution.iloc[family, 0]
        except:
            day = solution[family]

        choice = 9
        for i in range(1, 10):
            if initial_data.iloc[family, i] == day:
                choice = i - 1
                break
        if choice > 0:
            families_cost[int(day)] += get_cost_by_choice(family_size)[1][choice-1]
        family += 1

    return families_cost


def get_total_accounting_cost(daily_occupancy):
    # Calculate the accounting cost
    # The first day (day 100) is treated special
    daily_occupancy_fix = np.zeros(101)
    daily_occupancy_fix[100] = daily_occupancy[99]
    for i in range(100):
        daily_occupancy_fix[i] = daily_occupancy[i]

    accounting_cost = (daily_occupancy_fix[days[0]] - 125.0) / 400.0 * daily_occupancy_fix[days[0]] ** (0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)

    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy_fix[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy_fix[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy_fix[day] - 125.0) / 400.0 * daily_occupancy_fix[day] ** (0.5 + diff / 50.0))
        yesterday_count = today_count

    return accounting_cost


def get_accounting_cost_per_day(daily_occupancy):
    # Calculate the accounting cost
    # The first day (day 100) is treated special
    daily_accounting_cost = np.zeros(101)
    daily_occupancy_fix = np.zeros(101)
    daily_occupancy_fix[100] = daily_occupancy[99]
    for i in range(100):
        daily_occupancy_fix[i] = daily_occupancy[i]

    accounting_cost = (daily_occupancy_fix[days[0]] - 125.0) / 400.0 * daily_occupancy_fix[days[0]] ** (0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)

    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy_fix[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy_fix[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy_fix[day] - 125.0) / 400.0 * daily_occupancy_fix[day] ** (0.5 + diff / 50.0))
        daily_accounting_cost[day] = max(0, (daily_occupancy_fix[day] - 125.0) / 400.0 * daily_occupancy_fix[day] ** (0.5 + diff / 50.0))
        yesterday_count = today_count

    return daily_accounting_cost


if __name__ == "__main__":
    initial_data = return_family_data()

    solution = load_solution_data('submission_on_mip_lst_69267.78950400071.csv')

    # daily_load_idea = load_occupancy_data()
    # accounting_cost = get_total_accounting_cost(daily_load_idea)

    daily_load = compute_daily_load_2(solution, initial_data)
    choice_cost = np.sum(get_choice_cost(solution, initial_data))
    accounting_cost = get_total_accounting_cost(daily_load)

    acc_cost = get_accounting_cost_per_day(daily_load)
    ch_cost_per_fam = get_choice_cost_per_family(solution, initial_data)

    choices = calculate_choice_id_per_family(solution, initial_data)

    print('statistics per choice ' + str(np.unique(choices, return_counts=True)))

    print('accounting cost :' + str(accounting_cost))
    print('choice_cost :' + str(choice_cost))

    print('total_cost' + str(choice_cost+accounting_cost))

