import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


# # TODO: to be refactored & moved to analyze solution
# def plot_family_wishes(choice_min, choice_max, df):
#     days_vs_people = np.zeros(101)
#
#     row = 0
#     while row < df.shape[0]:
#         nr_of_people = df.iloc[row, 11]
#         for choice in range(choice_min, choice_max):
#             day = df.iloc[row, choice+1]
#             days_vs_people[day] = days_vs_people[day] + nr_of_people
#         row = row + 1
#         if (row > 4997) :
#             print(row)
#
#     plt.figure(figsize=(34, 50))
#     newdf = pd.DataFrame(days_vs_people)
#     ax = sns.barplot(x=newdf.index, y=np.concatenate(newdf.values))
#
#     ax.set_ylim(0, 1.1 * 3000)
#     plt.xlabel('Family Size', fontsize=14)
#     plt.ylabel('Count', fontsize=14)
#     plt.title('Family Size Distribution', fontsize=20)
#     plt.show()


# TODO: to be moved to load_santa_data
def return_family_data():
    data_load = santa.SantaDataLoad()
    df = data_load.load_family_initial_data("D:\\jde\\projects\\santas_workshop_2019\\santadata\\")
    return df


# TODO: to be moved to load_santa_data
def return_family_sizes(df):
    return list(df['n_people'])


def search_for_exchange(family_choices_ids, family_choices_days, choice_cost, accounting_old, people_count, family_sizes, df, negative_threashold):
    if negative_threashold > 0 :
        print('running with negative threshold ' + str(negative_threashold))
    new_exchange_found_ids = family_choices_ids
    new_exchange_found_days = family_choices_days
    new_cost = choice_cost
    new_accounting = accounting_old
    new_people_count = people_count
    for family_id, choice in enumerate(new_exchange_found_ids):
        print('checking family : ' + str(family_id))
        family_choices = np.zeros(10)
        for column in range(1, 11):
            day = df.iloc[family_id, column]
            family_choices[column-1] = day
        for family_id_2, choice_2 in enumerate(new_exchange_found_ids):
            family_2_choices = np.zeros(10)
            if family_id == family_id_2:
                continue
            for column_2 in range(1, 11):
                day_2 = df.iloc[family_id_2, column]
                family_2_choices[column - 1] = day_2
            if new_exchange_found_days[family_id_2] in family_choices and new_exchange_found_days[family_id] in family_2_choices:
                if new_exchange_found_days[family_id] != new_exchange_found_days[family_id_2]:
                    family_choices_ids_copy = new_exchange_found_ids.copy()
                    family_choices_days_copy = new_exchange_found_days.copy()
                    family_choices_ids_copy[family_id] = choice_2
                    family_choices_ids_copy[family_id_2] = choice
                    family_choices_days_copy[family_id] = new_exchange_found_days[family_id_2]
                    family_choices_days_copy[family_id_2] = new_exchange_found_days[family_id]
                    people_count_copy = new_people_count.copy()

                    new_choice_cost = an_solution.get_choice_cost(family_choices_days_copy, df)

                    people_count_copy[int(new_exchange_found_days[family_id])] -= family_sizes[family_id]
                    people_count_copy[int(new_exchange_found_days[family_id])] += family_sizes[family_id_2]
                    people_count_copy[int(new_exchange_found_days[family_id_2])] -= family_sizes[family_id_2]
                    people_count_copy[int(new_exchange_found_days[family_id_2])] += family_sizes[family_id]
                    accounting_new = an_solution.get_total_accounting_cost(people_count_copy)

                    day_old_ok = 125 <= people_count_copy[int(new_exchange_found_days[family_id])] <= 300
                    day_new_ok = 125 <= people_count_copy[int(new_exchange_found_days[family_id_2])] <= 300

                    if not day_old_ok or not day_new_ok:
                        #print("move breaks day constraint")
                        continue

                    old_total_cost = np.sum(new_cost) + new_accounting
                    new_total_cost = np.sum(new_choice_cost) + accounting_new

                    should_accept_negatives = False
                    if (negative_threashold > 0):
                        if (new_total_cost - old_total_cost < negative_threashold):
                            should_accept_negatives = True
                            print('new_total_cost - old_total_cost : ' + str(new_total_cost - old_total_cost))

                    if ((new_total_cost < old_total_cost) and day_old_ok and day_new_ok and not should_accept_negatives) \
                            or (should_accept_negatives and new_total_cost >= old_total_cost) :
                        print("BINGO, fam id " + str(family_id) + ", choice cost is: " + str(np.sum(new_choice_cost)) +
                              " accounting cost is : " + str(accounting_new))
                        new_exchange_found_ids = family_choices_ids_copy
                        new_exchange_found_days = family_choices_days_copy
                        new_cost = new_choice_cost
                        new_accounting = accounting_new
                        new_people_count = people_count_copy

    return new_exchange_found_ids, new_exchange_found_days, new_cost, new_people_count, new_accounting


def search_for_move(family_choices_ids, family_choices_days, choice_cost, accounting_old, people_count, df, threashold):
    new_exchange_found_ids = family_choices_ids
    new_exchange_found_days = family_choices_days
    new_cost = choice_cost
    new_accounting = accounting_old
    new_people_count = people_count
    for family_id, choice in enumerate(new_exchange_found_ids):
        print('checking family : ' + str(family_id))
        for choice_id in range(0, 6):
            family_choices_ids_copy = new_exchange_found_ids.copy()
            family_choices_days_copy = new_exchange_found_days.copy()
            people_count_copy = new_people_count.copy()
            family_nr_of_people = df.iloc[family_id, 11]
            if choice_id != new_exchange_found_ids[family_id]:
                new_day = df.iloc[family_id, choice_id+1]
                old_day = new_exchange_found_days[family_id]
                family_choices_ids_copy[family_id] = choice_id
                family_choices_days_copy[family_id] = new_day
                people_count_copy[int(old_day)] -= family_nr_of_people
                people_count_copy[int(new_day)] += family_nr_of_people

                day_old_ok = 125 <= people_count_copy[int(old_day)] <= 300
                day_new_ok = 125 <= people_count_copy[int(new_day)] <= 300

                if not day_old_ok or not day_new_ok:
                    #print("move breaks day constraint")
                    continue

                new_choice_cost = an_solution.get_choice_cost(family_choices_days_copy, df)
                accounting_new = an_solution.get_total_accounting_cost(people_count_copy)

                old_total_cost = np.sum(new_cost) + new_accounting
                new_total_cost = np.sum(new_choice_cost) + accounting_new

                should_accept_negatives = False
                if (threashold > 0):
                    if (new_total_cost - old_total_cost < threashold):
                        should_accept_negatives = True
                    print ('new_total_cost - old_total_cost : ' + str(new_total_cost - old_total_cost))
                if ((new_total_cost < old_total_cost) and day_old_ok and day_new_ok and should_accept_negatives is False) \
                        or (should_accept_negatives and new_total_cost >= old_total_cost):
                    print("BINGO, fam id " + str(family_id) + ", choice cost is: " + str(np.sum(new_choice_cost)) +
                          " accounting cost is : " + str(accounting_new))
                    new_exchange_found_ids = family_choices_ids_copy
                    new_exchange_found_days = family_choices_days_copy
                    new_cost = new_choice_cost
                    new_accounting = accounting_new
                    new_people_count = people_count_copy
                # else:
                    # print("fam id " + str(family_id) + ", choice cost is: " + str(np.sum(computed_cost)) +
                    #       " accounting cost is : " + str(accounting_new) + " trial choice_id is " + str(choice_id)
                    #       + "family_choices_ids[family_id] is " + str(family_choices_ids[family_id]))

    return new_exchange_found_ids, new_exchange_found_days, new_cost, new_people_count, new_accounting


def return_family_data():
    data_load = santa.SantaDataLoad()
    df = data_load.load_family_initial_data("D:\\jde\\projects\\santas_workshop_2019\\santadata\\")
    return df


if __name__ == "__main__":

    solution = an_solution.load_solution_data('test_submission_stoc_71699_54_negative_82_3.csv')
    initial_data = return_family_data()

    days_load = an_solution.compute_daily_load(solution, initial_data)

    accounting_cost = an_solution.get_total_accounting_cost(days_load)
    choice_cost = an_solution.get_choice_cost(solution, initial_data)
    family_choices_ids = an_solution.calculate_choice_id_per_family(solution, initial_data)
    family_choices_days = np.array(solution['assigned_day'])

    print("Accounting cost is : " + str(accounting_cost))
    print("choice cost is : " + str(np.sum(choice_cost)))
    family_sizes = an_solution.return_family_sizes(initial_data)

    ## don't touch it is good to optimize the second metric mainly (once the proportion is set.
    # iteration = 0
    # accept_threshold = 0
    # while np.sum(choice_cost) > 69000 or iteration < 3:
    #     new_exchange_found, new_exchange_found_days, new_cost, people_count_copy, accounting_new\
    #         = search_for_exchange(family_choices_ids,
    #                             family_choices_days,
    #                             choice_cost,
    #                             accounting_cost,
    #                             days_load,
    #                             family_sizes,
    #                             initial_data,
    #                             accept_threshold)
    #
    #     if np.sum(choice_cost) == np.sum(new_cost):
    #         if (accept_threshold == 0):
    #             accept_threshold += 300
    #         accept_threshold += 100
    #     else:
    #         iteration = 0
    #         accept_threshold = 0
    #         data_load = santa.SantaDataLoad()
    #         all_cost = np.sum(new_cost) + accounting_new
    #         data_load.save_submission('D:\\jde\\projects\\santas_workshop_2019\\santadata\\', str(all_cost),
    #                                   new_exchange_found_days)
    #     iteration += 1
    #
    #
    #     family_choices_ids = new_exchange_found
    #     family_choices_days = new_exchange_found_days
    #     choice_cost = new_cost
    #     accounting = accounting_new
    #     total_cost = choice_cost + accounting_cost
    #     people_count = people_count_copy


    # optimize for first metric (choice cost)
    iteration = 0
    accept_threshold = 0
    while np.sum(choice_cost) > 69000 or iteration < 3:
        print('iteration index : ' + str(iteration))
        new_exchange_found, new_exchange_found_days, new_cost, people_count_copy, accounting_new\
              = search_for_move(family_choices_ids,
                                  family_choices_days,
                                  choice_cost,
                                  accounting_cost,
                                  days_load,
                                  initial_data,
                                  accept_threshold)

        if np.sum(choice_cost) == np.sum(new_cost):
            accept_threshold += 10
        else:
            accept_threshold = 0
            iteration = 0
            data_load = santa.SantaDataLoad()
            all_cost = np.sum(new_cost) + accounting_new
            data_load.save_submission('D:\\jde\\projects\\santas_workshop_2019\\santadata\\', str(all_cost),
                                      new_exchange_found_days)

        iteration += 1

        family_choices_ids = new_exchange_found
        family_choices_days = new_exchange_found_days
        choice_cost = new_cost
        accounting_cost = accounting_new
        total_cost = np.sum(choice_cost) + accounting_cost
        days_load = people_count_copy

        print("Accounting cost is : " + str(accounting_cost))
        print("choice cost is : " + str(np.sum(choice_cost)))




