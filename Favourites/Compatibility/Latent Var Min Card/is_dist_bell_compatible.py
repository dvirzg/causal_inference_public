import numpy as np
import itertools
import sympy as sp
import gurobipy as gp
from gurobipy import GRB

# Define the four functions
## Bin(INT) -> Bin(INT)
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

def generate_top(u):
    f_u, g_u = all_strategies[u]
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



if __name__ == "__main__":
    # declaring Gurobi variable W_i w/ i in {0,1,...,15}, and declaring v
    # m = gp.Model()
    # W = m.addVars(16, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="W")
    # m.update()
    global is_compatible
    # dist_ABXY = {}


    # # # get P_AB|XY from P_ABXY
    # def P_AB_giv_XY(A,B, X, Y):
    #     P_ABXY = sum([dist_ABXY[(a,b,x,y)] for a,b,x,y in dist_ABXY if a==A and b==B and x==X and y==Y])
    #     P_AB = sum([dist_ABXY[(a,b,x,y)] for a,b,x,y in dist_ABXY if a==A and b==B])
    #     return P_ABXY/P_AB

    # internal_gurobi_expr = sum([W[i]*generate_top(i) for i in range(16)])
    
    # m.addConstr(W.sum() == 1, "sum(W_i) == 1")
    # for x,y, a,b in itertools.product(range(2), repeat=4):
    #     m.addConstr(internal_gurobi_expr[x,y,a,b] == P_AB_giv_XY(a,b,x,y), f"P({a},{b}|{x},{y}) == {internal_gurobi_expr[x,y,a,b]} == {P_AB_giv_XY(a,b,x,y)}")
    # m.update()
    # m.optimize()

    # if m.status == 2: # GRB.OPTIMAL
    #     is_compatible = True
    # else: 
    #     is_compatible = False
    is_compatible = False
