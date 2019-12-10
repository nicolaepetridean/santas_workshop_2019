import random
import numpy as np
import pandas as pd
from numba import njit
from itertools import product
import matplotlib.pylab as plt
from ortools.linear_solver import pywraplp
from analyze.analyze_solution import load_solution_data

def get_penalty(n, choice):
    penalty = None
    if choice == 0:
        penalty = 0
    elif choice == 1:
        penalty = 50
    elif choice == 2:
        penalty = 50 + 9 * n
    elif choice == 3:
        penalty = 100 + 9 * n
    elif choice == 4:
        penalty = 200 + 9 * n
    elif choice == 5:
        penalty = 200 + 18 * n
    elif choice == 6:
        penalty = 300 + 18 * n
    elif choice == 7:
        penalty = 300 + 36 * n
    elif choice == 8:
        penalty = 400 + 36 * n
    elif choice == 9:
        penalty = 500 + 36 * n + 199 * n
    else:
        penalty = 500 + 36 * n + 398 * n
    return penalty

def GetPreferenceCostMatrix(data):
    cost_matrix = np.zeros((N_FAMILIES, N_DAYS), dtype=np.int64)
    for i in range(N_FAMILIES):
        desired = data.values[i, :-1]
        cost_matrix[i, :] = get_penalty(FAMILY_SIZE[i], 10)
        for j, day in enumerate(desired):
            cost_matrix[i, day-1] = get_penalty(FAMILY_SIZE[i], j)
    return cost_matrix


def GetAccountingCostMatrix():
    ac = np.zeros((1000, 1000), dtype=np.float64)
    for n in range(ac.shape[0]):
        for n_p1 in range(ac.shape[1]):
            diff = abs(n - n_p1)
            ac[n, n_p1] = max(0, (n - 125) / 400 * n**(0.5 + diff / 50.0))
    return ac

# cost_function, etc.

# preference cost
@njit(fastmath=True)
def pcost(prediction):
    daily_occupancy = np.zeros(N_DAYS+1, dtype=np.int64)
    penalty = 0
    for (i, p) in enumerate(prediction):
        n = FAMILY_SIZE[i]
        penalty += PCOSTM[i, p]
        daily_occupancy[p] += n
    return penalty, daily_occupancy


# accounting cost
@njit(fastmath=True)
def acost(daily_occupancy):
    accounting_cost = 0
    n_out_of_range = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_p1 = daily_occupancy[day + 1]
        n    = daily_occupancy[day]
        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)
        accounting_cost += ACOSTM[n, n_p1]
    return accounting_cost, n_out_of_range

@njit(fastmath=True)
def acostd(daily_occupancy):
    accounting_cost = np.zeros(N_DAYS, dtype=np.float64)
    n_out_of_range = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_p1 = daily_occupancy[day + 1]
        n    = daily_occupancy[day]
        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)
        accounting_cost[day] = ACOSTM[n, n_p1]
    return accounting_cost, n_out_of_range

@njit(fastmath=True)
def pcostd(prediction):
    daily_occupancy = np.zeros(N_DAYS+1, dtype=np.int64)
    penalty = np.empty_like(prediction)
    for (i, p) in enumerate(prediction):
        n = FAMILY_SIZE[i]
        penalty[i] = PCOSTM[i, p]
        daily_occupancy[p] += n
    return penalty, daily_occupancy


@njit(fastmath=True)
def cost_stats(prediction):
    penalty, daily_occupancy = pcostd(prediction)
    accounting_cost, n_out_of_range = acostd(daily_occupancy)
    return penalty, accounting_cost, n_out_of_range, daily_occupancy[:-1]


@njit(fastmath=True)
def cost_function(prediction):
    penalty, daily_occupancy = pcost(prediction)
    accounting_cost, n_out_of_range = acost(daily_occupancy)
    return penalty + accounting_cost + n_out_of_range*100000000


# fixMinOccupancy, fixMaxOccupancy + helpers

@njit(fastmath=True)
def cost_function_(prediction):
    penalty, daily_occupancy = pcost(prediction)
    accounting_cost, n_out_of_range = acost(daily_occupancy)
    return penalty + accounting_cost, n_out_of_range


@njit(fastmath=True)
def findAnotherDay4Fam(prediction, fam, occupancy):
    old_day = prediction[fam]
    best_cost = np.inf
    best_day = fam
    n = FAMILY_SIZE[fam]

    daysrange = list(range(0, old_day)) + list(range(old_day + 1, N_DAYS))
    for day in daysrange:
        prediction[fam] = day
        new_cost, _ = cost_function_(prediction)

        if (new_cost < best_cost) and (occupancy[day] + n <= MAX_OCCUPANCY):
            best_cost = new_cost
            best_day = day

    prediction[fam] = old_day
    return best_day, best_cost


@njit(fastmath=True)
def bestFamAdd(prediction, day, occupancy):
    best_cost = np.inf
    best_fam = prediction[day]
    for fam in np.where(prediction!=day)[0]:
        old_day = prediction[fam]
        prediction[fam] = day
        new_cost, _ = cost_function_(prediction)
        prediction[fam] = old_day
        n = FAMILY_SIZE[fam]
        if (new_cost<best_cost) and (occupancy[old_day]-n>=MIN_OCCUPANCY):
            best_cost = new_cost
            best_fam = fam
    return best_fam


@njit(fastmath=True)
def bestFamRemoval(prediction, day, occupancy):
    best_cost = np.inf
    best_day = day

    for fam in np.where(prediction == day)[0]:
        new_day, new_cost = findAnotherDay4Fam(prediction, fam, occupancy)
        if new_cost < best_cost:
            best_cost = new_cost
            best_fam = fam
            best_day = new_day

    return best_fam, best_day


@njit(fastmath=True)
def fixMaxOccupancy(prediction):
    penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)

    for day in np.where(occupancy > MAX_OCCUPANCY)[0]:
        while occupancy[day] > MAX_OCCUPANCY:
            fam, new_day = bestFamRemoval(prediction, day, occupancy)
            prediction[fam] = new_day
            penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)


@njit(fastmath=True)
def fixMinOccupancy(prediction):
    penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)

    for day in np.where(occupancy < MIN_OCCUPANCY)[0]:
        while occupancy[day] < MIN_OCCUPANCY:
            fam = bestFamAdd(prediction, day, occupancy)
            prediction[fam] = day
            penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)


# swappers

def findBetterDay4Family(pred):
    fobs = np.argsort(FAMILY_SIZE)
    score = cost_function(pred)
    original_score = np.inf

    while original_score > score:
        original_score = score
        for family_id in fobs:
            for pick in range(10):
                day = DESIRED[family_id, pick]
                oldvalue = pred[family_id]
                pred[family_id] = day
                new_score = cost_function(pred)
                if new_score < score:
                    score = new_score
                else:
                    pred[family_id] = oldvalue

        print(score, end='\r')
    print(score)


def stochastic_product_search(top_k_jump, top_k, fam_size, original,
                              verbose=1000, verbose2=50000,
                              n_iter=500, random_state=2019):
    """
    original (np.array): The original day assignments.

    At every iterations, randomly sample fam_size families. Then, given their top_k
    choices, compute the Cartesian product of the families' choices, and compute the
    score for each of those top_k^fam_size products.
    """

    best = original.copy()
    best_score = cost_function(best)

    np.random.seed(random_state)
    min_obtained_score = 100000

    for i in range(n_iter):
        fam_indices = np.random.choice(range(DESIRED.shape[0]), size=fam_size)
        #fam_indices = np.append(fam_indices, [1415])
        changes = np.array(list(product(*DESIRED[fam_indices, top_k_jump:top_k].tolist())))

        for change in changes:
            new = best.copy()
            new[fam_indices] = change

            new_score = cost_function(new)

            if new_score < min_obtained_score:
                min_obtained_score = new_score

            if (new_score <= best_score) or (0 < best_score - new_score < 25):
                best_score = new_score
                best = new

        if verbose and i % verbose == 0:
            print(f"Iteration #{i}: Best score is {best_score:.2f}      ", end='\r')

        if verbose2 and i % verbose2 == 0:
            print(f"Iteration #{i}: Best score is {best_score:.2f}      ")
            print(f"Iteration #{i}: new score is {new_score:.2f}      ")
            print(f"Iteration #{i}: min score score is {min_obtained_score:.2f}      ")
            print(f"Iteration #{i}: family indices are {str(fam_indices)}      ")

    print(f"Final best score is {best_score:.2f}")
    return best


def solveSantaLP():
    S = pywraplp.Solver('SolveAssignmentProblem', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    x = {}
    for i in range(N_FAMILIES):
        for j in range(N_DAYS):
            x[i, j] = S.BoolVar('x[%i,%i]' % (i, j))

    daily_occupancy = [S.Sum([x[i, j] * FAMILY_SIZE[i] for i in range(N_FAMILIES)])
                       for j in range(N_DAYS)]

    family_presence = [S.Sum([x[i, j] for j in range(N_DAYS)])
                       for i in range(N_FAMILIES)]

    # Objective
    preference_cost = S.Sum([PCOSTM[i, j] * x[i, j] for i in range(N_FAMILIES)
                             for j in range(N_DAYS)])

    S.Minimize(preference_cost)

    # Constraints
    for j in range(N_DAYS - 1):
        S.Add(daily_occupancy[j] - daily_occupancy[j + 1] <= 32)
        S.Add(daily_occupancy[j + 1] - daily_occupancy[j] <= 31)

    for i in range(N_FAMILIES):
        S.Add(family_presence[i] == 1)

    for j in range(N_DAYS):
        S.Add(daily_occupancy[j] >= MIN_OCCUPANCY)
        S.Add(daily_occupancy[j] <= MAX_OCCUPANCY)

    res = S.Solve()

    resdict = {0: 'OPTIMAL', 1: 'FEASIBLE', 2: 'INFEASIBLE', 3: 'UNBOUNDED',
               4: 'ABNORMAL', 5: 'MODEL_INVALID', 6: 'NOT_SOLVED'}

    print('Result:', resdict[res])

    l = []
    for i in range(N_FAMILIES):
        for j in range(N_DAYS):
            l.append((i, j, x[i, j].solution_value()))

    df = pd.DataFrame(l, columns=['family_id', 'day', 'n'])

    if len(df) != N_FAMILIES:
        df = df.sort_values(['family_id', 'n']).drop_duplicates('family_id', keep='last')

    return df.day.values

if __name__ == '__main__' :
    N_DAYS = 100
    N_FAMILIES = 5000
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125

    data = pd.read_csv('D:\\jde\\projects\\santas_workshop_2019\\santadata\\family_data.csv', index_col='family_id')

    FAMILY_SIZE = data.n_people.values
    DESIRED     = data.values[:, :-1] - 1
    PCOSTM = GetPreferenceCostMatrix(data) # Preference cost matrix
    ACOSTM = GetAccountingCostMatrix()     # Accounting cost matrix

    prediction = load_solution_data('submission_stoc_71613_jumptop0_828.csv')

    prediction = prediction['assigned_day'].to_numpy()

    # prediction = solveSantaLP()
    # penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)
    # print(penalty.sum(), accounting_cost.sum(), n_out_of_range, occupancy.min(), occupancy.max())
    # #
    # penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)
    # print(penalty.sum(), accounting_cost.sum(), n_out_of_range, occupancy.min(), occupancy.max())
    #
    #
    # fixMinOccupancy(prediction)
    # fixMaxOccupancy(prediction)
    prediction = prediction - 1
    penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)
    print('{}, {:.0f}'.format(penalty.sum(), accounting_cost.sum()))

    iteration = 1

    fam_size_out = 4
    n_iter = 500000
    while fam_size_out > 1:
        final = stochastic_product_search(
                top_k_jump=0,
                top_k=3,
                fam_size=fam_size_out,
                original=prediction,
                n_iter=n_iter,
                verbose=500,
                verbose2=2500,
                random_state=2019
                )

        prediction = final

        sub = pd.DataFrame(range(N_FAMILIES), columns=['family_id'])
        sub['assigned_day'] = final + 1
        sub.to_csv('D:\\jde\\projects\\santas_workshop_2019\\santadata\\submission_stoc_71613_jumptop0_43' + str(fam_size_out) + '.csv', index=False)

        fam_size_out -= 1

    # final_1 = stochastic_product_search(
    #         top_k=2,
    #         fam_size=8,
    #         original=final,
    #         n_iter=50000,
    #         verbose=1000,
    #         verbose2=50000,
    #         random_state=2019
    #         )
    #
    #
    # final_2 = stochastic_product_search(
    #         top_k=2,
    #         fam_size=9,
    #         original=final_1,
    #         n_iter=50000,
    #         verbose=1000,
    #         verbose2=100000,
    #         random_state=2019
    #         )
