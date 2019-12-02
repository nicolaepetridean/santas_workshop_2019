import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


if __name__ == "__main__":
    print(get_cost_by_choice(2))
    print(get_cost_by_choice(3))
    print(get_cost_by_choice(4))
    print(get_cost_by_choice(5))
    print(get_cost_by_choice(6))
    print(get_cost_by_choice(7))
    print(get_cost_by_choice(8))

    # plot_family_sizes()

    # Test it with several test cases
    # Constant 210, cost should be 3.08/day
    people_count = np.concatenate((125 + np.zeros(51), 295 + np.zeros(50)), axis=0)
    # Alternate 210 +/-25 = 185, 235. Should be ~ 674/day
    # people_count = (210 + 25*np.cos(np.pi*np.arange(0,102))).astype(int)
    # Alternate 210 +/-50 = 160, 260. Should be ~ 194,000/day
    ##people_count = (210 + 50*np.cos(np.pi*np.arange(0,102))).astype(int)
    #
    accounting = accounting_cost(people_count)

    print("Accounting cost is : " + str(accounting))