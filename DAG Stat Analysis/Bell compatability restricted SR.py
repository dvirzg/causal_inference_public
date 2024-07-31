import gurobipy as gp
import numpy as np

"""
Our goal is to evaluate how close to a given correlation we can get under a cardinality restriction.
"""

def bell_compatability(distribution: np.ndarray, card_L: int):
    (card_X, card_Y, card_A, card_B) = distribution.shape
    with gp.Env(empty=True) as env:
        env.setParam('LogToConsole', True)  # To supress output
        env.start()
        with gp.Model("qcp", env=env) as m:
            m.params.NonConvex = 2  # Using quadratic equality constraints.
            m.params.LogFile = "solver_output.txt"
            m.setParam('PreSparsify', 1)
            m.setParam('PreQLinearize', 1)
            m.setParam('OptimalityTol', 0.01)
            m.setParam('PreDepRow', 1)
            m.setParam('Symmetry', 2)
            # m.setParam('Heuristics', 0)
            # m.setParam('RINS', 0)
            m.setParam('MIPFocus', 3)
            m.setParam('MinRelNodes', 0)
            m.setParam('ZeroObjNodes', 0)
            m.setParam('ImproveStartGap', 1)

            response_A = m.addMVar((card_X, 1, card_A, 1, card_L), lb=0, name="p[A|XL]")
            response_B = m.addMVar((1, card_Y, 1, card_B, card_L), lb=0, name="p[B|YL]")
            response_L = m.addMVar((1, 1, 1, 1, card_L), lb=0, name="p[L]")
            m.addConstr(response_A.sum(axis=2) == 1)
            m.addConstr(response_B.sum(axis=3) == 1)
            local_fraction = m.addVar(name="local_fraction", lb=0, ub=1)
            m.addConstr(response_L.sum(axis=4) == local_fraction)  # Upper bound for local fraction
            # visibility = m.addVar(name="visibility", lb=0, ub=1)
            # m.addConstr(response_L.sum(axis=4) == 1)



            prod_AL = m.addMVar((card_X, 1, card_A, 1, card_L), lb=0, ub=1)
            m.addConstr(prod_AL == response_A * response_L)
            unobservable_probs = m.addMVar((card_X, card_Y, card_A, card_B, card_L), lb=0, ub=1, name="Q")
            observable_probs = m.addMVar((card_X, card_Y, card_A, card_B), name="P_compat")
            m.addConstr(unobservable_probs == prod_AL * response_B)
            m.addConstr(observable_probs == unobservable_probs.sum(axis=4))
            m.addConstr(distribution >= observable_probs) # for local fraction
            # m.addConstr(distribution*visibility <= observable_probs)
            m.setObjective(local_fraction, sense=gp.GRB.MAXIMIZE)
            # m.setObjective(visibility, sense=gp.GRB.MAXIMIZE)


            # distance_from_target = m.addVar(name="distance")
            # slack_vector = m.addMVar((card_X, card_Y, card_A, card_B), name="slack", lb=-1, ub=1)
            # m.addConstr(slack_vector == observable_probs - distribution)
            # m.addGenConstrNorm(distance_from_target, slack_vector.reshape(-1), 2, "normconstr")
            # m.setObjective(distance_from_target, sense=gp.GRB.MINIMIZE)

            m.update()
            m.optimize()
            record_to_preserve = dict()
            if m.getAttr("SolCount"):
                print('\x1b[6;30;41m' + f'Classical fraction at cardinality {card_L}:' + '\x1b[6;30;42m' + f' {m.getObjective().getValue()}' + '\x1b[0m')
                # print('\x1b[6;30;41m' + f'Visibility at cardinality {card_L}:' + '\x1b[6;30;42m' + f' {m.getObjective().getValue()}' + '\x1b[0m')
                # print('\x1b[6;30;41m' + f'Smallest L2 norm at cardinality {card_L}:' + '\x1b[6;30;42m' + f' {m.getObjective().getValue()}' + '\x1b[0m')
                for var in observable_probs.reshape(-1).tolist():
                    record_to_preserve[var.VarName] = var.X
        m.dispose()
    env.dispose()
    return record_to_preserve

d=2
example_dist = np.full((d, d, 2, 2), 1/4, dtype=float)
for i in range(d):
    example_dist[i, i, 0, 0] = 1 / 2
    example_dist[i, i, 1, 1] = 1 / 2
    example_dist[i, i, 0, 1] = 0
    example_dist[i, i, 1, 0] = 0

d=2
prime_list = [2,3,5,7,11,13,17,19,23,29]
# prime_list = np.broadcast_to(2,10)
# for this (3,3,2,2) 3 settings two binary outcomes - min card 7 for total classical fraction 
relative_primes_dist = np.zeros((d, d, 2, 2), dtype=float)
for (x, y, a, b) in np.ndindex(relative_primes_dist.shape):
    px = prime_list[x]
    py = prime_list[y]
    x_factor = (1+(a*px)-(2*a))/px
    y_factor = (1+(b*py)-(2*b))/py
    if x==y:
        if a==b:
            relative_primes_dist[x, y, a, b] = x_factor
        else:
            relative_primes_dist[x, y, a, b] = 0
    else:
        relative_primes_dist[x, y, a, b] = x_factor*y_factor

# for x in range(d):
#    for y in range(x+1):
#        print(f"Distribution when x={x} and {y}=y:")
#        print(relative_primes_dist[x, y])


# print(bell_compatability(relative_primes_dist, 7))
# print("\n\nhere:\n",relative_primes_dist)

arr = np.array([[[[0.20042701, 0.45280667],
         [0.15579778, 0.29675916]],

        [[0.49962199, 0.24724233],
         [0.30833322, 0.16737184]]],


       [[[0.08116999, 0.18771233],
         [0.12579922, 0.34375984]],

        [[0.21878101, 0.11223867],
         [0.41006978, 0.19210916]]]])

arr2 = np.zeros((2,2,2,2))
for a,b,x,y in np.ndindex(2,2,2,2):
    arr2[x,y,a,b] = arr[a,b,x,y]
print(bell_compatability(arr2, 2))