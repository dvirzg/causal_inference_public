
# ## Interuption DAG Task


# ...


import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np


card_X = 2
card_Y = 2
card_Z = 2


observable_probs = np.arange(card_X * card_Y * card_Z)
observable_probs = observable_probs / observable_probs.sum()
observable_probs = observable_probs.reshape(card_X, card_Y, card_Z)

dist_XYZ = {}
# defining a p distribution based on cardinality of A, B, C
for x, y, z in itertools.product(range(card_X), range(card_Y), range(card_Z)):
    print(observable_probs[0,0,0])
    prob = observable_probs[x, y, z]
    dist_XYZ[(x, y, z)] = prob


# given distribution P_XYZ get P_XY|Z:
dist_XY_giv_Z = {}
for x, y, z in itertools.product(range(card_X), range(card_Y), range(card_Z)):
    prob = dist_XYZ[(x, y, z)]
    dist_XY_giv_Z[(x, y, z)] = prob / observable_probs[:, :, z].sum()


# distribution feasible?


dist_XYZ


print(sum(dist_XYZ.values()) == 1)


dist_XY_giv_Z


# $P_Z(Z)$


# P_Z(z) = sum_{X,Y} P_XYZ(x,y,z)
def P_Z(z):
    return sum([dist_XYZ[(X, Y, z)] for X in range(card_X) for Y in range(card_Y)])


# $P_\lambda (\lambda), \space P_{A|\lambda} (A, \lambda), \space P_{C|B,\lambda} (C, B, \lambda)$


m = gp.Model()
m.Params.LogToConsole = 0
# m.setParam('OutputFlag', 2)


# variables
P_Xb_Y_giv_Z_Xs = m.addMVar(shape = (card_X, card_Y, card_Z, card_X), vtype=GRB.CONTINUOUS, name="P_Xb_Y_giv_Z_Xs", lb=0, ub=1)
# P_Y_Z_giv_Xs = m.addMVar(shape = (card_Y, card_Z, card_X), vtype=GRB.CONTINUOUS, name="P_Y_Z_giv_Xs", lb=0, ub=1)
P_Y_giv_Z_Xs = m.addMVar(shape = (card_Y, card_Z, card_X), vtype=GRB.CONTINUOUS, name="P_Y_giv_Z_Xs", lb=0, ub=1)
P_Xb_giv_Z_Xs = m.addMVar(shape = (card_X, card_Z, card_X), vtype=GRB.CONTINUOUS, name="P_Xb_giv_Z_Xs", lb=0, ub=1)
P_Y_giv_Xs = m.addMVar(shape = (card_Y, card_X), vtype=GRB.CONTINUOUS, name="P_Y_giv_Xs", lb=0, ub=1)
P_Xb_giv_Z = m.addMVar(shape = (card_X, card_Z), vtype=GRB.CONTINUOUS, name="P_Xb_giv_Z", lb=0, ub=1)
# P_Z_giv_Xs = m.addMVar(shape = (card_Z, card_X), vtype=GRB.CONTINUOUS, name="P_Z_giv_Xs", lb=0, ub=1)

# consistency constraints
for x, y, z in itertools.product(range(card_X), range(card_Y), range(card_Z)):
    m.addConstr(P_Xb_Y_giv_Z_Xs[x,y,z,x] == dist_XY_giv_Z[(x, y, z)])


### marginalization constraints:

# sum over Y of P(Xb,Y|Z,X#) == P(Xb|Z,X#) / also #=P(Xb|Z) = P(X|Z)
for xb, xs, z in itertools.product(range(card_X), range(card_X), range(card_Z)):
    m.addConstr(P_Xb_giv_Z_Xs[xb,z,xs] == sum([P_Xb_Y_giv_Z_Xs[xb,y,z,xs] for y in range(card_Y)]))
    m.addConstr(P_Xb_giv_Z_Xs[xb,z,xs] == P_Xb_giv_Z[xb,z]) #INDEP OF Xb from Xs

# P_Y|ZX# = P_Y|X#
for x, y, z in itertools.product(range(card_X), range(card_Y), range(card_Z)):
    m.addConstr(P_Y_giv_Z_Xs[y,z,x] == sum([P_Xb_Y_giv_Z_Xs[xb,y,z,x] for xb in range(card_X)]))
    m.addConstr(P_Y_giv_Z_Xs[y,z,x] == P_Y_giv_Xs[y,x]) #INDEP OF Y from Z


"""
### independance constraints:

# ## Y d-indep of Z | X#
# for y, x in itertools.product(range(card_Y), range(card_X)):
#     # P(Y, Z=0|X#) = P(Y, Z=1|X#)
#     m.addConstr(P_Y_Z_giv_Xs[y,0,x] == P_Y_Z_giv_Xs[y,1,x])    

#     # P(Y_Z|X#) = P(Y|X#)P(Z|X#) = P(Y|X#)P(Z)
#     temp1 = P_Y_Z_giv_Xs[y,z,x]
#     m.addConstr(temp1 == P_Y_giv_Xs[y,x]*P_Z_giv_Xs[z,x])
#     m.addConstr(temp1 == P_Y_giv_Xs[y,x]*P_Z(z)) #Z d-indep of X#:
    
#     # P(Y|Z=0, X#) = P(Y|Z=1, X#)
#     m.addConstr(P_Y_giv_Z_Xs[y,0,x] == P_Y_giv_Z_Xs[y,1,x]) 
      
# ## Xb d-indep of X# | Z:
# for x, z in itertools.product(range(card_X), range(card_Z)):
#     # P(Xb|Z, X#=0) = P(Xb|Z, X#=1)
#     m.addConstr(P_Xb_giv_Z_Xs[x,z,0] == P_Xb_giv_Z_Xs[x,z,1])
    

# sum to one constraints

# # P_Y_giv_Xs when summed over y should be 1
# for x in range(card_X):
#     m.addConstr(sum([P_Y_giv_Xs[y,x] for y in range(card_Y)]) == 1)

# # P_Xb_giv_Z_Xs when summed over xb should be 1
# for z, xs in itertools.product(range(card_Z), range(card_X)):
#         m.addConstr(sum([P_Xb_giv_Z_Xs[xb,z,xs] for xb in range(card_X)]) == 1)



# do-conditional P_Y|do(X#) = P_Y|X#
# P_Y|do(X#) = sum_{l} P_l*P(Y|Z,X# P_l) # uneeded

# # find P_Y|X# from P_Xb_Y_giv_Z_Xs
# for y, xs in itertools.product(range(card_Y), range(card_X)):
#     m.addConstr(
#         P_Y_giv_Xs[y, xs] == sum(P_Xb_Y_giv_Z_Xs[xb, y, z, xs] * P_Z(z) for xb in range(card_X) for z in range(card_Z)),
#         name=f"Marginalization_{y}_{xs}")
# m.update()
"""


m.update()


# optomize for P_YdoX#
# P_YdoX# = P_Y_giv_Xs
def P_YdoXs(y,x):
    m.setObjective(P_Y_giv_Xs[y,x], GRB.MINIMIZE)
    m.optimize()
    min_val = P_Y_giv_Xs[y,x].X.item()

    m.setObjective(P_Y_giv_Xs[y,x], GRB.MAXIMIZE)
    m.optimize()
    max_val = P_Y_giv_Xs[y,x].X.item()

    print("\nmin value: ", min_val)
    print("max value: ", max_val)
    print("distance:", max_val - min_val)


# P_YdoXs(1,1)


#m.computeIIS()
#m.write("model.ilp")


