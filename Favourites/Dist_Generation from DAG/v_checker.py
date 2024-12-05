# %%
import numpy as np
import itertools
import sympy as sp

# Define the four functions

## Bin(INT), Bin(INT) -> Bin(INT)
def identity(x):
    return x

def swap(x):
    return 1-x

def to0(_):
    return 0

def to1(_):
    return 1


basic_strategies = [identity, swap, to0, to1]
all_strategies = list(itertools.product(basic_strategies, repeat=2))

# %%
import argparse
parser=argparse.ArgumentParser(description="v value")
parser.add_argument("v_val")
args=parser.parse_args()
v=args.v_val
print('v = ',v)

# %%
def generate_top(u):
    f_u, g_u = all_strategies[u]
    print(f"f_{u} = {f_u.__name__} \ng_{u} = {g_u.__name__}\n")

    # value 1 at point(a,b) = (f_u(x),g_u(y)), zero everywhere else
    def middle_entry(x,y):
        # value 1 at (f_u(x),g_u(y)) zero everywhere else
        middle = np.zeros((2,2), dtype=int)
        middle[f_u(x), g_u(y)] = 1
        return middle

    top_level = np.array([
                [middle_entry(0,0), middle_entry(1,0)], 
                [middle_entry(0,1), middle_entry(1,1)]
                ]) 

    return top_level


# u=5
# print("top level for strategy u =", u)
# generate_top(u)

# %%
# make it symbolic
tokens = [f"w{i}" for i in range(16)]
symb = sp.symbols(" ".join(tokens))

# %%

# add strategy matrices
summed_strategies = sum([symb[i]*generate_top(i) for i in range(16)])

# %%
summed_strategies

# %% [markdown]
# Found summed strategies matrices for all U!

# %%
# declaring Gurobi variable W_i with each lb=0 and ub=1
import gurobipy as gp
from gurobipy import GRB
m = gp.Model()
W = m.addVars(16, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="W")
m.update()

# %%
dist_ABXY = {
    (0, 0, 0, 0): 0.0,
    (0, 0, 0, 1): 0.008333333333333333,
    (0, 0, 1, 0): 0.016666666666666666,
    (0, 0, 1, 1): 0.025,
    (0, 1, 0, 0): 0.03333333333333333,
    (0, 1, 0, 1): 0.041666666666666664,
    (0, 1, 1, 0): 0.05,
    (0, 1, 1, 1): 0.058333333333333334,
    (1, 0, 0, 0): 0.06666666666666667,
    (1, 0, 0, 1): 0.075,
    (1, 0, 1, 0): 0.08333333333333333,
    (1, 0, 1, 1): 0.09166666666666666,
    (1, 1, 0, 0): 0.1,
    (1, 1, 0, 1): 0.10833333333333334,
    (1, 1, 1, 0): 0.11666666666666667,
    (1, 1, 1, 1): 0.125}
sum(dist_ABXY.values()) == 1

# get P_AB|XY from P_ABXY
def P_AB_giv_XY(A,B, X, Y):
    P_ABXY = sum([dist_ABXY[(a,b,x,y)] for a,b,x,y in dist_ABXY if a==A and b==B and x==X and y==Y])
    P_AB = sum([dist_ABXY[(a,b,x,y)] for a,b,x,y in dist_ABXY if a==A and b==B])
    return P_ABXY/P_AB


# dist_AB_given_XY_2 = {
#     (0,0,0,0): 0.5,
#     (1,1,0,0): 0.5,
#     (1,0,0,0): 0,
#     (0,1,0,0): 0,
#     (0,0,0,1): 0.25,
#     (1,1,0,1): 0.25,
#     (0,1,0,1): 0.25,
#     (1,0,0,1): 0.25,
#     (0,0,1,0): 0.5,
#     (1,1,1,0): 0.5,
#     (0,1,1,0): 0,
#     (1,0,1,0): 0,
#     (0,0,1,1): 0.25,
#     (1,1,1,1): 0.25,
#     (0,1,1,1): 0,
#     (1,0,1,1): 0.25}

# # get P_AB|XY from P_ABXY
# def P_AB_giv_XY(A,B, X, Y):
#     return dist_AB_given_XY_2[A,B,X,Y]

# %%
np.kron(0,0)

# %%
# P(AB|XY)

# v=0 # works!
# v=1 # infeasible
# v=0.000000000001
def P_AB_giv_XY(A,B, X, Y):
    if (A % 2) & B == 0 and X*Y == 0:
        return v*0.5 + (1-v)/4
    else:
        return v*(0.5*np.kron((A % 2) & B, X*Y)) + (1-v)/4


# %%
# summed_strategies[x,y][a,b]       # [x,y][a,b]
# P_AB_giv_XY(a,b,x,y)              # [a,b][x,y]

# %%
sym_to_gurobi = {symb[i]: W[i] for i in range(16)}
for x,y, a,b in itertools.product(range(2), repeat=4):
    # summed_strategies returns SymPy expression
    sympy_expr = summed_strategies[x,y][a,b]

    # Convert the SymPy expression to a Gurobi expression
    gurobi_expr = sum(sym_to_gurobi[sym] * coef for sym, coef in sympy_expr.as_coefficients_dict().items())

    # Add the constraint to the model
    m.addConstr(gurobi_expr == P_AB_giv_XY(a,b,x,y), f"P({a},{b}|{x},{y}) == {gurobi_expr} == {P_AB_giv_XY(a,b,x,y)}")


# %%
m.update()
constr = m.getConstrs()

# %% [markdown]
# ## SymPy Expression

# %%
# summed_strategies[x,y][a,b]
# print("P(a,b|x,y)")

for x,y, a,b in itertools.product(range(2), repeat=4):
     print(f"P({a},{b}|{x},{y}) = ", summed_strategies[x,y][a,b], " = ", P_AB_giv_XY(a,b,x,y))

# %% [markdown]
# ## Gurobi Constraints

# %%
for cons in constr:
    print(cons)

# %%
# set objective, find all W_i
obj = sum(W[i] for i in range(16))
m.setObjective(obj, GRB.MAXIMIZE)
m.update()

m.optimize()

# %%
print("***********************")
# for i in range(16):
#     print(f"W_{i} = {W[i].x}")


