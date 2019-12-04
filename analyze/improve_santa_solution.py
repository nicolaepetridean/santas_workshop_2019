import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import analyze.load_santa_data as santa
import analyze.analyze_solution as an_solution


NDAYS = 100
NFAMS = 5000
MAX_PPL = 300
MIN_PPL = 125

# The family preference cost parameters
PENALTY_CONST = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
PENALTY_PPL = [0, 0, 9, 9, 9, 18, 18, 36, 36, 199+36, 398+36]

# The seed is set once here at beginning of notebook.
RANDOM_SEED = 127
np.random.seed(RANDOM_SEED)


# TODO: to be refactored & moved to analyze solution
def plot_family_wishes(choice_min, choice_max, df):
    days_vs_people = np.zeros(101)

    row = 0
    while row < df.shape[0]:
        nr_of_people = df.iloc[row, 11]
        for choice in range(choice_min, choice_max):
            day = df.iloc[row, choice+1]
            days_vs_people[day] = days_vs_people[day] + nr_of_people
        row = row + 1
        if (row > 4997) :
            print(row)

    plt.figure(figsize=(34, 50))
    newdf = pd.DataFrame(days_vs_people)
    ax = sns.barplot(x=newdf.index, y=np.concatenate(newdf.values))

    ax.set_ylim(0, 1.1 * 3000)
    plt.xlabel('Family Size', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Family Size Distribution', fontsize=20)
    plt.show()


# TODO: to be moved to load_santa_data
def return_family_data():
    data_load = santa.SantaDataLoad()
    df = data_load.load_family_initial_data("/Users/nicolaepetridean/jde/projects/santas_workshop_2019/santadata/")
    return df


# TODO: to be moved to load_santa_data
def return_family_sizes(df):
    return list(df['n_people'])


def search_for_exchange(family_choices_ids, family_choices_days, choice_cost, accounting_old, people_count, family_sizes, df):
    new_exchange_found_ids = None
    new_exchange_found_days = None
    new_cost = 0
    for family_id, choice in enumerate(family_choices_ids):
        if choice > 0:
            for family_id_2, choice_2 in enumerate(family_choices_ids):
                if choice_2 > 0 and family_id != family_id_2 and family_choices_days[family_id] != family_choices_days[family_id_2]:
                    family_choices_ids_copy = family_choices_ids.copy()
                    family_choices_days_copy = family_choices_days.copy()
                    family_choices_ids_copy[family_id] = choice_2
                    family_choices_ids_copy[family_id_2] = choice
                    family_choices_days_copy[family_id] = family_choices_days[family_id_2]
                    family_choices_days_copy[family_id_2] = family_choices_days[family_id]
                    people_count_copy = people_count.copy()

                    computed_cost = compute_choice_cost(family_choices_ids_copy, df)
                    people_count_copy[int(family_choices_days[family_id])] -= family_sizes[family_id]
                    people_count_copy[int(family_choices_days[family_id])] += family_sizes[family_id_2]
                    people_count_copy[int(family_choices_days[family_id_2])] -= family_sizes[family_id_2]
                    people_count_copy[int(family_choices_days[family_id_2])] += family_sizes[family_id]
                    accounting_new = an_solution.accounting_cost(people_count_copy)

                    gain_cost = choice_cost - computed_cost
                    gain_accounting = accounting_old - accounting_new

                    switch_cond = gain_cost > 0 and (gain_accounting > 0 or gain_cost > (accounting_new - accounting_old))
                    switch_cond_2 = gain_accounting > 0 and (gain_cost > 0 or gain_accounting > (computed_cost - choice_cost))

                    if switch_cond or switch_cond_2:
                        print("BINGO, gain_cost is: " + str(computed_cost) + " accounting cost is : " + str(accounting_new))
                        new_exchange_found_ids = family_choices_ids_copy
                        new_exchange_found_days = family_choices_days_copy
                        new_cost = computed_cost
                        break
        if new_exchange_found_ids is not None:
            break

    return new_exchange_found_ids, new_exchange_found_days, new_cost, people_count_copy, accounting_new


def search_for_move(family_choices_ids, family_choices_days, choice_cost, accounting_old, people_count, family_sizes, df):
    new_exchange_found_ids = None
    new_exchange_found_days = None
    new_cost = 0
    new_accounting = 0
    new_people_count = None
    for family_id, choice in enumerate(family_choices_ids):
        if choice > 0:

            for choice_id in range(1, 5):
                family_choices_ids_copy = family_choices_ids.copy()
                family_choices_days_copy = family_choices_days.copy()
                people_count_copy = people_count.copy()
                family_nr_of_people = df.iloc[family_id, 11]
                if choice_id-1 != family_choices_ids[family_id]:
                    new_day = df.iloc[family_id, choice_id]
                    old_day = family_choices_days[family_id]
                    family_choices_ids_copy[family_id] = choice_id-1
                    family_choices_days_copy[family_id] = new_day
                    people_count_copy[int(old_day)] -= family_nr_of_people
                    people_count_copy[int(new_day)] -= family_nr_of_people

                    day_old_ok = 125 < people_count_copy[int(old_day)] < 300
                    day_new_ok = 125 < people_count_copy[int(new_day)] < 300

                    computed_cost = an_solution.get_accounting_cost(people_count_copy, df)
                    accounting_new = an_solution.get_accounting_cost(people_count_copy)

                    gain_cost = choice_cost - computed_cost
                    gain_accounting = accounting_old - accounting_new

                    switch_cond = gain_cost > 0 and (gain_accounting > 0 or gain_cost > (accounting_new - accounting_old))
                    switch_cond_2 = gain_accounting > 0 and (gain_cost > 0 or gain_accounting > (computed_cost - choice_cost))
                    if (switch_cond or switch_cond_2) and day_old_ok and day_new_ok:
                        print("BINGO, fam id " + str(family_id) + ", choice cost is: " + str(computed_cost) + " accounting cost is : " + str(accounting_new))
                        new_exchange_found_ids = family_choices_ids_copy
                        new_exchange_found_days = family_choices_days_copy
                        new_cost = computed_cost
                        new_accounting = accounting_new
                        new_people_count = people_count_copy
                        break

        if new_exchange_found_ids is not None:
            break

    return new_exchange_found_ids, new_exchange_found_days, new_cost, new_people_count, new_accounting


if __name__ == "__main__":

    df = return_family_data()

    accounting = an_solution.accounting_cost(people_count)

    choice_cost = compute_choice_cost(family_choices_ids, df)
    print("Accounting cost is : " + str(accounting))
    print("choice cost is : " + str(np.sum(choice_cost)))
    family_sizes = return_family_sizes(df)

    ## don't touch it is good to optimize the second metric mainly (once the proportion is set.
    # while choice_cost > 200000:
    #     new_exchange_found, new_exchange_found_days, new_cost, people_count_copy, accounting_new\
    #         = search_for_exchange(family_choices_ids,
    #                             family_choices_days,
    #                             choice_cost,
    #                             accounting,
    #                             people_count,
    #                             family_sizes,
    #                             df)
    #     family_choices_ids = new_exchange_found
    #     family_choices_days = new_exchange_found_days
    #     choice_cost = new_cost
    #     accounting = accounting_new
    #     people_count = people_count_copy

    iteration = 0
    while choice_cost > 200000 or iteration > 10000:
        new_exchange_found, new_exchange_found_days, new_cost, people_count_copy, accounting_new\
              = search_for_move(family_choices_ids,
                                  family_choices_days,
                                  choice_cost,
                                  accounting,
                                  people_count,
                                  family_sizes,
                                  df)
        family_choices_ids = new_exchange_found
        family_choices_days = new_exchange_found_days
        choice_cost = new_cost
        accounting = accounting_new
        people_count = people_count_copy

        data_load = santa.SantaDataLoad()
        data_load.save_submission('/Users/nicolaepetridean/jde/projects/santas_workshop_2019/santadata/', family_choices_days)

        iteration += 1


    print("Accounting cost is : " + str(accounting))
    print("choice cost is : " + str(np.sum(choice_cost)))


