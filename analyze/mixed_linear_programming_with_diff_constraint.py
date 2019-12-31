import random
import numpy as np
import pandas as pd
from numba import njit
from itertools import product
import matplotlib.pylab as plt
from ortools.linear_solver import pywraplp
from analyze.analyze_solution import load_solution_data, return_family_data, compute_daily_load

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
    return penalty + n_out_of_range*100000000, accounting_cost


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
    for fam in np.where(prediction != day)[0]:
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


def solveSantaLP(existing_occupancy, existing_prediction):
    S = pywraplp.Solver('SolveAssignmentProblem', pywraplp.Solver.BOP_INTEGER_PROGRAMMING)

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
    for i in range(N_FAMILIES):
        S.Add(family_presence[i] == 1)

    for j in range(N_DAYS):
        minim = max(existing_occupancy[j + 1], 125)
        maxim = min(existing_occupancy[j + 1], 300)
        if j not in [36, 43, 50, 57, 64, 71, 78, 85, 92, 99]:
            minim = max(existing_occupancy[j+1]-1, 125)
            maxim = min(existing_occupancy[j+1]+1, 300)
        S.Add(daily_occupancy[j] <= maxim)
        S.Add(daily_occupancy[j] >= minim)

    # for d in range(N_DAYS - 1):
    #     S.Add(daily_occupancy[d]-daily_occupancy[d+1] <= existing_occupancy[d] - existing_occupancy[d+1] + 1)
        #S.Add(daily_occupancy[d+1]-daily_occupancy[d] <= existing_occupancy[d+1] - existing_occupancy[d] + 1)
        # else:
        #     S.Add(daily_occupancy[d+1]-daily_occupancy[d] <= (existing_occupancy[d+1]-existing_occupancy[d] + 1))

    S.EnableOutput()
    S.set_time_limit(800*3600)

    valid_solution = []
    for family in range(N_FAMILIES):
        valid_solution.extend([False for i in range(N_DAYS)])
        valid_solution[family * N_DAYS + existing_prediction['assigned_day'][family] - 1] = True

    S.SetHint([x[i, j] for i in range(N_FAMILIES) for j in range(N_DAYS)], valid_solution)

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

    data = pd.read_csv('/Users/nicolaepetridean/jde/projects/santas_workshop_2019/santadata/family_data.csv', index_col='family_id')

    FAMILY_SIZE = data.n_people.values
    DESIRED     = data.values[:, :-1] - 1
    PCOSTM = GetPreferenceCostMatrix(data) # Preference cost matrix
    ACOSTM = GetAccountingCostMatrix()     # Accounting cost matrix

    initial_data = return_family_data()
    existing_prediction = load_solution_data('submission_on_jump_69469.3540022548.csv')
    daily_load = compute_daily_load(existing_prediction, initial_data)

    prediction = solveSantaLP(daily_load, existing_prediction)
    penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)
    print(penalty.sum(), accounting_cost.sum(), n_out_of_range, occupancy.min(), occupancy.max())
    #
    fixMinOccupancy(prediction)
    fixMaxOccupancy(prediction)
    penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)

    sub = pd.DataFrame(range(N_FAMILIES), columns=['family_id'])
    sub['assigned_day'] = prediction+1
    sub.to_csv('/Users/nicolaepetridean/jde/projects/santas_workshop_2019/santadata/try_mixed_with_diff.csv', index=False)

    print('GAHGS {}, {:.0f}'.format(penalty.sum(), accounting_cost.sum()))



