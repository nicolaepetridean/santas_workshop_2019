from ortools.linear_solver import pywraplp

MAX_BEST_CHOICE = 5
NUM_SWAP = 2500
NUM_SECONDS = 1800
NUM_THREADS = 4
for _ in range(20):
    solver = pywraplp.Solver('Optimization preference cost', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    daily_occupancy = get_daily_occupancy(assigned_days).astype(float)
    fids = np.random.choice(range(NUMBER_FAMILIES), NUM_SWAP, replace=False)
    PCOSTM, B = {}, {}
    for fid in range(NUMBER_FAMILIES):
        if fid in fids:
            for i in range(MAX_BEST_CHOICE):
                PCOSTM[fid, DESIRED[fid][i] - 1] = COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]
                B[fid, DESIRED[fid][i] - 1] = solver.BoolVar('')
        else:
            daily_occupancy[assigned_days[fid] - 1] -= N_PEOPLE[fid]

    solver.set_time_limit(NUM_SECONDS * NUM_THREADS * 1000)
    solver.SetNumThreads(NUM_THREADS)

    for day in range(NUMBER_DAYS):
        if daily_occupancy[day]:
            solver.Add(solver.Sum([N_PEOPLE[fid] * B[fid, day] for fid in range(NUMBER_FAMILIES) if (fid, day) in B]) ==
                       daily_occupancy[day])

    for fid in fids:
        solver.Add(solver.Sum(B[fid, day] for day in range(NUMBER_DAYS) if (fid, day) in B) == 1)

    solver.Minimize(solver.Sum(PCOSTM[fid, day] * B[fid, day] for fid, day in B))
    sol = solver.Solve()

    status = ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ABNORMAL', 'MODEL_INVALID', 'NOT_SOLVED']
    if status[sol] in ['OPTIMAL', 'FEASIBLE']:
        tmp = assigned_days.copy()
        for fid, day in B:
            if B[fid, day].solution_value() > 0.5:
                tmp[fid] = day + 1
        if cost_function(tmp)[2] < cost_function(assigned_days)[2]:
            assigned_days = tmp
            submission['assigned_day'] = assigned_days
            submission.to_csv('submission.csv', index=False)
        print('Result:', status[sol], cost_function(tmp))
    else:
        print('Result:', status[sol])