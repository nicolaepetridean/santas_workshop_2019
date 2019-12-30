%%time
from ortools.graph import pywrapgraph
for num_members in range(2, 9): # Families have minimum 2 and maximum 8 members
    daily_occupancy = get_daily_occupancy(assigned_days)
    fids = np.where(N_PEOPLE == num_members)[0]

    PCOSTM = {}
    for fid in range(NUMBER_FAMILIES):
        if fid in fids:
            for i in range(MAX_BEST_CHOICE):
                PCOSTM[fid, DESIRED[fid][i]-1] = COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]
        else:
            daily_occupancy[assigned_days[fid]-1] -= N_PEOPLE[fid]

    offset = fids.shape[0]
    solver = pywrapgraph.SimpleMinCostFlow()
    for day in range(NUMBER_DAYS):
        solver.SetNodeSupply(offset+day, int(daily_occupancy[day]//num_members))

    for i in range(offset):
        fid = fids[i]
        solver.SetNodeSupply(i, -1)
        for j in range(MAX_BEST_CHOICE):
            day = DESIRED[fid][j]-1
            solver.AddArcWithCapacityAndUnitCost(int(offset+day), i, 1, int(PCOSTM[fid, day]))
    solver.SolveMaxFlowWithMinCost()

    for i in range(solver.NumArcs()):
        if solver.Flow(i) > 0:
            assigned_days[fids[solver.Head(i)]] = solver.Tail(i) - offset + 1
    print(cost_function(assigned_days))