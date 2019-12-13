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


# def plot_daily_load(days_load):
#     plt.figure(figsize=(34, 50))
#     newdf = pd.DataFrame(days_load)
#     ax = sns.barplot(x=newdf.index, y=np.concatenate(newdf.values))
#     ax.set_ylim(0, 1.1 * 1000)
#     plt.xlabel('day', fontsize=14)
#     plt.ylabel('Count', fontsize=14)
#     plt.title('Day Load', fontsize=20)
#     plt.show()
#
#     return days_load


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


# def plot_choice_cost(days_cost):
#     days_cost = days_cost[1:]
#     print(" sum of all cost is :" + str(np.sum(days_cost)))
#     print(" min of all days cost is :" + str(np.min(days_cost)))
#     print(" max of all days cost is :" + str(np.max(days_cost)))
#     plt.figure(figsize=(34, 50))
#     newdf = pd.DataFrame(days_cost)
#     ax = sns.barplot(x=newdf.index, y=np.concatenate(newdf.values))
#     ax.set_ylim(0, 1.1 * 10000)
#     plt.xlabel('Family Size', fontsize=14)
#     plt.xticks(range(0, 100, 5))
#     plt.ylabel('Count', fontsize=14)
#     plt.title('Family Size Distribution', fontsize=20)
#     plt.show()
#
#     return np.sum(days_cost)


# def plot_accounting_cost(days_cost):
#     days_cost = days_cost[1:]
#     print(" sum of all cost is :" + str(np.sum(days_cost)))
#     print(" min of all days cost is :" + str(np.min(days_cost)))
#     print(" max of all days cost is :" + str(np.max(days_cost)))
#     plt.figure(figsize=(34, 50))
#     newdf = pd.DataFrame(days_cost)
#     ax = sns.barplot(x=newdf.index, y=np.concatenate(newdf.values))
#     ax.set_ylim(0, 1.1 * 800)
#     plt.xlabel('day', fontsize=14)
#     plt.ylabel('cost', fontsize=14)
#     plt.xticks(range(0, 100, 5))
#     plt.title('Daily accounting Distribution', fontsize=20)
#     plt.show()
#
#     return np.sum(days_cost)


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

    # solution = load_solution_data('submission_76101.75179796087.csv')
    solution = load_solution_data('submission_71393.75_BASE.csv')
    # solution = load_solution_data('sample_submission_output_test.csv')
    # solution = load_solution_data('sample_submission_output55_76448_submit.csv')

    #daily_load = plot_daily_load(compute_daily_load(solution, initial_data))
    daily_load = compute_daily_load(solution, initial_data)

    #choice_cost = plot_choice_cost(get_choice_cost(solution, initial_data))
    choice_cost = np.sum(get_choice_cost(solution, initial_data))

    accounting_cost = get_total_accounting_cost(daily_load)

    acc_cost = get_accounting_cost_per_day(daily_load)
    # plot_accounting_cost(acc_cost)

    choices = calculate_choice_id_per_family(solution, initial_data)


    print('statistics per choice ' + str(np.unique(choices, return_counts=True)))

    print('accounting cost :' + str(accounting_cost))
    print('choice_cost :' + str(choice_cost))

    print('total_cost' + str(choice_cost+accounting_cost))

