import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import analyze.load_santa_data as santa


data = pd.read_csv('/Users/nicolaepetridean/jde/projects/titanic/try/santadata/family_data.csv', index_col='family_id')

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
    df = data_load.load_file("/Users/nicolaepetridean/jde/projects/titanic/try/santadata/")
    return df


def load_solution_data(solution_file = "sample_submission_output_fix.csv"):
    data_load = santa.SantaDataLoad()
    df = data_load.load_solution_file("/Users/nicolaepetridean/jde/projects/titanic/try/santadata/" + solution_file)
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


def calculate_accounting_cost(daily_occupancy, days):
    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]] - 125.0) / 400.0 * daily_occupancy[days[0]] ** (0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)

    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day] - 125.0) / 400.0 * daily_occupancy[day] ** (0.5 + diff / 50.0))
        yesterday_count = today_count

    return accounting_cost


def plot_daily_load(solution, initial_data):
    days_load = np.zeros(101)
    row = 0
    while row < solution.shape[0]:
        day = solution.iloc[row, 1]
        n_people = initial_data.iloc[row, 11]
        days_load[day] += n_people
        row += 1
    days_load = days_load[1:]
    print(" sum of all people is :" + str(np.sum(days_load)))
    print(" min of all days is :" + str(np.min(days_load)))
    print(" max of all days is :" + str(np.max(days_load)))
    plt.figure(figsize=(34, 50))
    newdf = pd.DataFrame(days_load)
    ax = sns.barplot(x=newdf.index, y=np.concatenate(newdf.values))
    ax.set_ylim(0, 1.1 * 3000)
    plt.xlabel('Family Size', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Family Size Distribution', fontsize=20)
    plt.show()

    return days_load


def plot_and_return_choice_cost(solution, initial_data):
    days_cost = np.zeros(101)
    row = 0
    while row < solution.shape[0]:
        family_size = initial_data.iloc[row, 11]
        day = solution.iloc[row, 1]

        choice = 9
        for i in range(1, 7):
            if initial_data.iloc[row, i] == day:
                choice = i - 1
        if choice > 0:
            days_cost[day] += get_cost_by_choice(family_size)[1][choice-1]
        row += 1

    days_cost = days_cost[1:]
    print(" sum of all cost is :" + str(np.sum(days_cost)))
    print(" min of all days cost is :" + str(np.min(days_cost)))
    print(" max of all days cost is :" + str(np.max(days_cost)))
    plt.figure(figsize=(34, 50))
    newdf = pd.DataFrame(days_cost)
    ax = sns.barplot(x=newdf.index, y=np.concatenate(newdf.values))
    ax.set_ylim(0, 1.1 * 10000)
    plt.xlabel('Family Size', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Family Size Distribution', fontsize=20)
    plt.show()

    return np.sum(days_cost)


def cost_function(prediction):
    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k: 0 for k in days}

    # Looping over each family; d is the day for each family f
    for f, d in enumerate(prediction):
        if f == 0:
            continue

        # Using our lookup dictionaries to make simpler variable names
        n = family_size_dict[f]
        choice_0 = choice_dict['choice_0'][f]
        choice_1 = choice_dict['choice_1'][f]
        choice_2 = choice_dict['choice_2'][f]
        choice_3 = choice_dict['choice_3'][f]
        choice_4 = choice_dict['choice_4'][f]
        choice_5 = choice_dict['choice_5'][f]
        choice_6 = choice_dict['choice_6'][f]
        choice_7 = choice_dict['choice_7'][f]
        choice_8 = choice_dict['choice_8'][f]
        choice_9 = choice_dict['choice_9'][f]

        # add the family member count to the daily occupancy
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d == choice_0:
            penalty += 0
        elif d == choice_1:
            penalty += 50
        elif d == choice_2:
            penalty += 50 + 9 * n
        elif d == choice_3:
            penalty += 100 + 9 * n
        elif d == choice_4:
            penalty += 200 + 9 * n
        elif d == choice_5:
            penalty += 200 + 18 * n
        elif d == choice_6:
            penalty += 300 + 18 * n
        elif d == choice_7:
            penalty += 300 + 36 * n
        elif d == choice_8:
            penalty += 400 + 36 * n
        elif d == choice_9:
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for _, v in daily_occupancy.items():
        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):
            penalty += 100000000

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]] - 125.0) / 400.0 * daily_occupancy[days[0]] ** (0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)

    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day] - 125.0) / 400.0 * daily_occupancy[day] ** (0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty


if __name__ == "__main__":
    initial_data = return_family_data()

    solution = load_solution_data('submission_672254.0276683343.csv')

    # plot_daily_load(solution, initial_data)
    #
    # plot_and_return_choice_cost(solution, initial_data)

    total_cost = cost_function(solution)

    print('total_cost' + str(solution))

