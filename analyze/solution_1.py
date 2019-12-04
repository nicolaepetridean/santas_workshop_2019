import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from array import array
import analyze.load_santa_data as santa


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

# Define the accounting cost function
def accounting_cost(people_count):
    # people_count[iday] is an array of the number of people each day,
    # valid for iday=1 to NDAYS (iday=0 not used).
    total_cost = 0.0
    ppl_yester = people_count[NDAYS]
    for iday in range(NDAYS,0,-1):
        ppl_today = people_count[iday]
        ppl_delta = np.abs(ppl_today - ppl_yester)
        day_cost = (ppl_today - 125)*(ppl_today**(0.5+ppl_delta/50.0))/400.0
        total_cost += day_cost
        ##print("Day {}: delta = {}, $ {}".format(iday, ppl_delta, int(day_cost)))
        # save for tomorrow
        ppl_yester = people_count[iday]
    print("Total accounting cost: {:.2f}.  Ave costs:  {:.2f}/day,  {:.2f}/family".format(
        total_cost,total_cost/NDAYS,total_cost/NFAMS))
    return total_cost

def plot_family_sizes():
    data_load = santa.SantaDataLoad()

    df = data_load.load_file("/Users/nicolaepetridean/jde/projects/titanic/try/santadata/")
    family_size = df['n_people'].value_counts().sort_index()

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=family_size.index, y=family_size.values)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}\n({p.get_height() / sum(family_size) * 100:.1f}%)',
                    xy=(p.get_x() + p.get_width() / 2., p.get_height()), ha='center', xytext=(0, 5),
                    textcoords='offset points')

    ax.set_ylim(0, 1.1 * max(family_size))
    plt.xlabel('Family Size', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Family Size Distribution', fontsize=20)
    plt.show()


def plot_family_wishes(choice_min, choice_max):
    days_vs_people = np.zeros(101)
    data_load = santa.SantaDataLoad()

    df = data_load.load_file("/Users/nicolaepetridean/jde/projects/titanic/try/santadata/")

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


def choose_family_wishes(choice_max):
    days_vs_people = np.zeros(101)
    data_load = santa.SantaDataLoad()

    df = data_load.load_file("/Users/nicolaepetridean/jde/projects/titanic/try/santadata/")

    family_id = 0
    choice_id_per_family = np.zeros(5000)
    choisen_day_per_family = np.zeros(5000)
    while family_id < df.shape[0]:
        nr_of_people = df.iloc[family_id, 11]
        chosen_day = -1

        # choice_zero
        choice_1 = df.iloc[family_id, 1]
        if days_vs_people[choice_1] + nr_of_people < 126:
            chosen_day = choice_1
            choice_id_per_family[family_id] = 0
        else:
            # choice_one
            choice_2 = df.iloc[family_id, 2]
            choice_2_people = days_vs_people[choice_2] + nr_of_people
            if choice_2_people < 126:
                chosen_day = choice_2
                choice_id_per_family[family_id] = 1
            else:
                #choose alternative choices
                if choice_max >= 2:
                    minim = 1000
                    for choice in range(2, choice_max):
                        choice_alt = df.iloc[family_id, choice+1]
                        if (days_vs_people[choice_alt] < minim):
                            minim = days_vs_people[choice_alt]
                            if minim < 126:
                                chosen_day = choice_alt
                                choice_id_per_family[family_id] = choice
                    if chosen_day == -1:
                        choice_0 = df.iloc[family_id, 1]
                        ch_0_alt = days_vs_people[choice_0] + nr_of_people
                        choice_1 = df.iloc[family_id, 2]
                        ch_1_alt = days_vs_people[choice_1] + nr_of_people
                        if ch_0_alt < ch_1_alt:
                            chosen_day = choice_1
                            choice_id_per_family[family_id] = 0
                        else :
                            chosen_day = choice_0
                            choice_id_per_family[family_id] = 1


        days_vs_people[chosen_day] = days_vs_people[chosen_day] + nr_of_people
        choisen_day_per_family[family_id] = chosen_day

        family_id = family_id + 1

    print("validate the number of people is right " + str(np.sum(days_vs_people)))

    plt.figure(figsize=(34, 50))
    newdf = pd.DataFrame(days_vs_people)
    ax = sns.barplot(x=newdf.index, y=np.concatenate(newdf.values))

    ax.set_ylim(0, 1.1 * 3000)
    plt.xlabel('Family Size', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Family Size Distribution', fontsize=20)
    plt.show()

    return (days_vs_people, choice_id_per_family, choisen_day_per_family)

if __name__ == "__main__":
    print(get_cost_by_choice(2))
    # print(get_cost_by_choice(3))
    # print(get_cost_by_choice(4))
    # print(get_cost_by_choice(5))
    # print(get_cost_by_choice(6))
    # print(get_cost_by_choice(7))
    # print(get_cost_by_choice(8))

    # plot_family_sizes()

    # Test it with several test cases
    # Constant 210, cost should be 3.08/day
    # people_count = np.concatenate((295 + np.zeros(50), 125 + np.zeros(51)), axis=0)
    # Alternate 210 +/-25 = 185, 235. Should be ~ 674/day
    # people_count = (210 + 25*np.cos(np.pi*np.arange(0,102))).astype(int)
    # Alternate 210 +/-50 = 160, 260. Should be ~ 194,000/day
    ##people_count = (210 + 50*np.cos(np.pi*np.arange(0,102))).astype(int)
    #
    # accounting = accounting_cost(people_count)

    #plot_family_wishes(0,1)

    #plot_family_wishes(1, 2)

    # plot_family_wishes(0, 4)

    people_count, choice_id_per_family, choisen_day_per_family = choose_family_wishes(3)

    unique, counts = np.unique(choice_id_per_family, return_counts=True)

    print("counts per choice " + str(counts))

    accounting = accounting_cost(people_count)

    data_load = santa.SantaDataLoad()

    df = data_load.load_file("/Users/nicolaepetridean/jde/projects/titanic/try/santadata/")

    choice_cost = np.zeros(5000)
    row = 0
    while row < df.shape[0]:
        nr_of_people = df.iloc[row, 11]
        family_id = df.iloc[row, 0]
        final_choice = choice_id_per_family[row]
        if final_choice > 0:
            choice_cost = choice_cost + get_cost_by_choice(nr_of_people)[1][int(final_choice-1)]
        row = row + 1

    print("Accounting cost is : " + str(accounting))
    print("choice cost is : " + str(np.sum(choice_cost)))