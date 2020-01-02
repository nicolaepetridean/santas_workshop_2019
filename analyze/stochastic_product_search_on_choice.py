import random
import numpy as np
import pandas as pd
from numba import njit
from itertools import product
import matplotlib.pylab as plt
from ortools.linear_solver import pywraplp
from analyze.analyze_solution import load_solution_data, calculate_choice_id_per_family, return_family_data, get_choice_cost, compute_daily_load_2

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
    (init_choice_cost, init_accounting_cost) = cost_function(pred)
    score = init_choice_cost + init_accounting_cost
    original_score = np.inf

    while original_score > score:
        original_score = score
        for family_id in fobs:
            for pick in range(10):
                day = DESIRED[family_id, pick]
                oldvalue = pred[family_id]
                pred[family_id] = day
                (choice_cost, accounting_cost) = cost_function(pred)
                new_score = choice_cost + accounting_cost
                if new_score < score:
                    score = new_score
                else:
                    pred[family_id] = oldvalue

        print(score, end='\r')
    print(score)


def stochastic_product_search(top_k_jump, top_k, fam_size, original,
                              verbose=1000, verbose2=50000,
                              n_iter=500, random_state=2019, switch_candidates=[], initial_data=[]):
    """
    original (np.array): The original day assignments.

    At every iterations, randomly sample fam_size families. Then, given their top_k
    choices, compute the Cartesian product of the families' choices, and compute the
    score for each of those top_k^fam_size products.
    """

    best = original.copy()
    (best_choice_cost, best_accounting_cost) = cost_function(best)
    best_score = best_choice_cost + best_accounting_cost
    initial_score = best_score

    np.random.seed(random_state)
    min_obtained_score = 100000
    last_switch = 0

    lower_bound = 30
    upper_bound = 80

    for i in range(n_iter):
        last_switch += 1
        #candiates_fam_indices = np.random.choice((switch_candidates), size=1)
        # fam_size = np.random.choice(range(4, init_fam_size), size=1)[0]
        # top_k = np.random.choice(range(0, init_top_k), size=1)[0]
        np.random.seed(random_state + i)
        fam_indices = np.random.choice(range(DESIRED.shape[0]), size=fam_size)

        if n_iter < 5000:
            for id in fam_indices:
                if id in EXCLUDE:
                    fam_indices = np.delete(fam_indices, np.where(fam_indices == id))
        #fam_indices = np.append(fam_indices, candiates_fam_indices)
        changes = np.array(list(product(*DESIRED[fam_indices, top_k_jump:top_k].tolist())))

        for change in changes:
            new = best.copy()
            new[fam_indices] = change

            (new_choice_cost, new_accounting_cost) = cost_function(new)
            new_score = new_choice_cost + new_accounting_cost

            if (new_score < min_obtained_score):
                min_obtained_score = new_score

            if new_score < best_score:
                best_score = new_score
                best = new
                last_switch = 0
                print("New best score found : " + str(best_score))

                if best_score < 71342:
                    sub = pd.DataFrame(range(N_FAMILIES), columns=['family_id'])
                    sub['assigned_day'] = best + 1
                    sub.to_csv('D:\\jde\\projects\\santas_workshop_2019\\santadata\\new\\submission_' + str(
                        fam_size) + '_iter_' + str(i) + '_score_' + str(best_score) + '_.csv', index=False)

            else:
                if last_switch > 1300000:
                    if lower_bound < new_score - best_score < upper_bound:
                            best_score = new_score
                            best = new
                            print("JUMP. New best score found : " + str(best_score))
                            last_switch = 0

        if verbose and i % verbose == 0:
            print(f"Iteration #{i}: Best score is {best_score:.2f}      ", end='\r')

        if verbose2 and i % verbose2 == 0:
            print(f"Iteration #{i}: Best score is {best_score:.2f}      ")
            print(f"Iteration #{i}: Best iteration score is {min_obtained_score:.2f}      ")
            print(f"Iteration #{i}: new score is {new_score:.2f}      ")
            print(f"Iteration #{i}: family indices are {str(fam_indices)}      ")
            min_obtained_score = 100000

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

    data = pd.read_csv('/Users/nicolaepetridean/jde/projects/santas_workshop_2019/santadata/family_data.csv', index_col='family_id')

    FAMILY_SIZE = data.n_people.values
    DESIRED     = data.values[:, :-1] - 1 #data[data.n_people < 7].values[:, :-1] - 1
    EXCLUDE     = []
    PCOSTM = GetPreferenceCostMatrix(data) # Preference cost matrix
    ACOSTM = GetAccountingCostMatrix()     # Accounting cost matrix

    prediction = load_solution_data('submission_on_jump_69247.74599637784.csv')

    daily_load = compute_daily_load_2(prediction, data)
    mix_pool = []
    for item in range(prediction.shape[0]):
        assigned_day = prediction['assigned_day'][item]
        ch0 = data.iloc[item, 0]
        if data.iloc[item, 10] > 7:
            if daily_load[ch0] >= 298 and assigned_day == ch0:
                mix_pool.append(item)

    best_item_switch = None
    best_item_day = None
    best_item_cost = 0
    freed_day = 0

    for item in mix_pool:
        ex_day = prediction['assigned_day'][item]
        for ch in range(1, 5):
            candidate_day = data.iloc[item, ch]
            if daily_load[candidate_day] > 250:
                continue
            prediction['assigned_day'][item] = candidate_day
            ch_cost, acc_cost = cost_function(prediction['assigned_day'].to_numpy() - 1)
            if best_item_switch is None:
                best_item_switch = item
                best_item_day = candidate_day
                best_item_cost = ch_cost + int(acc_cost)
                freed_day = ex_day
            else:
                if ch_cost + int(acc_cost) < best_item_cost:
                    best_item_switch = item
                    best_item_day = candidate_day
                    best_item_cost = ch_cost + int(acc_cost)
                    freed_day = ex_day


            prediction['assigned_day'][item] = ex_day

    daily_load[freed_day] -= FAMILY_SIZE[best_item_switch]
    prediction['assigned_day'][best_item_switch] = best_item_day

    switches = []
    for item in range(data.shape[0]):
        if data.iloc[item, 0] == freed_day:
            if prediction['assigned_day'][item] != freed_day and FAMILY_SIZE[item] < 6:
                switches.append(item)

    start = 0
    sum = 0
    if len(switches) > 0:
        while daily_load[freed_day] < 299 and start < len(switches):
            if sum + FAMILY_SIZE[switches[start]] < FAMILY_SIZE[best_item_switch]:
                current_day = prediction['assigned_day'][switches[start]]
                if daily_load[current_day] - FAMILY_SIZE[switches[start]] >= 125:
                    daily_load[freed_day] += FAMILY_SIZE[switches[start]]
                    prediction['assigned_day'][switches[start]] = freed_day
                    sum += FAMILY_SIZE[switches[start]]
                    EXCLUDE.append(switches[start])
            start += 1

    print('moved family is ' + str(best_item_switch) + ' , to day : ' + str(best_item_day))
    print('sum of replaced families is ' + str(sum))
    print('found the following number items as candidates' + str(len(mix_pool)))

    EXCLUDE.append(best_item_switch)

    prediction = prediction['assigned_day'].to_numpy()
    prediction = prediction - 1
    penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)
    print('{}, {:.0f}'.format(penalty.sum(), accounting_cost.sum()))

    iteration = 1
    fam_size_out = 5
    n_iter = 5000000

    initial_data = return_family_data()
    while fam_size_out > 1:
        # compute non zero choices
        #switch_candidates = famillies
        final = stochastic_product_search(
                top_k_jump=0,
                top_k=3,
                fam_size=fam_size_out,
                original=prediction,
                n_iter=n_iter,
                verbose=1000,
                verbose2=1000,
                random_state=2037,
                switch_candidates=[],
                initial_data = initial_data
                )

        prediction = final

        sub = pd.DataFrame(range(N_FAMILIES), columns=['family_id'])
        sub['assigned_day'] = final + 1
        sub.to_csv('D:\\jde\\projects\\santas_workshop_2019\\santadata\\new\\submission_plusone_uniq_' + str(fam_size_out) + '.csv', index=False)
        sub['assigned_day'] = final
        sub.to_csv('D:\\jde\\projects\\santas_workshop_2019\\santadata\\new\\submission_uniq_' + str(fam_size_out) + '.csv',
                   index=False)

        fam_size_out -= 1
