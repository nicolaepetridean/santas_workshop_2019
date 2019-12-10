# santas_workshop_2019
personal solution for https://www.kaggle.com/c/santa-workshop-tour-2019/overview

Your task is to schedule the families to Santa's Workshop in a way that
minimizes the penalty cost to Santa (as described on the Evaluation page).

Each family has listed their top 10 preferences for the dates they'd
like to attend Santa's workshop tour. Dates are integer values
representing the days before Christmas, e.g., the value 1 represents
Dec 24, the value 2 represents Dec 23, etc. Each family also has a
number of people attending, n_people.

Every family must be scheduled for one and only one assigned_day


# ideas to try : do not forget
 -   try to move one family out of choices, even if second metric is
    blowing. and give some room to the first metric to improve.
 -   try to improve a very costy choice/day with another day that has cost.
 -   in the random family schoice generator jump over the families already on choice one. just to try to get some progress.-
 -   iterate on stochastic 8/2, 9/2 8/3 vs 5,5, 4/5 etc