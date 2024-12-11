import numpy as np
import gurobipy as gp
from gurobipy import GRB, norm

class BellCompatible:
    def __init__(self, card_A, card_B, card_X, card_Y, card_l, verbose=0): # settings and hidden common cause cardinality
        self.card_A = card_A
        self.card_B = card_B
        self.card_X = card_X
        self.card_Y = card_Y
        self.card_l = card_l
        self.verbose = verbose
        self.model = None

    def initialize_model(self):
        self.model = gp.Model("BellCompat")
        self.model.reset()
        self.model.setParam('OutputFlag', self.verbose)
        self.model.params.NonConvex = 2 

        # variables
        self.P_l = self.model.addMVar(self.card_l, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="P_l")
        self.P_A_giv_Xl = self.model.addMVar(shape=(self.card_A, self.card_X, self.card_l), vtype=GRB.CONTINUOUS, name="P(A|X,l)", lb=0, ub=1)
        self.P_B_giv_Yl = self.model.addMVar(shape=(self.card_B, self.card_Y, self.card_l), vtype=GRB.CONTINUOUS, name="P(B|Y,l)", lb=0, ub=1)
        self.prod = self.model.addMVar(shape=(self.card_B, self.card_l, self.card_Y), vtype=GRB.CONTINUOUS, name="P(B,l|Y)", lb=0, ub=1)
        self.model.update()

        # constraints
        for b, y, l in np.ndindex(self.card_B, self.card_Y, self.card_l):
            # bc can't multiply 3 vars in SEM constraint
            self.model.addConstr(self.prod[b, l, y] == self.P_B_giv_Yl[b, y, l] * self.P_l[l], name=f"P(B|Y,l) * P(l)")

        for x, l in np.ndindex(self.card_X, self.card_l):
            self.model.addConstr(sum([self.P_A_giv_Xl[a, x, l] for a in range(self.card_A)]) == 1, name=f'sum P(a|{x,l}) = 1')
        for y, l in np.ndindex(self.card_Y, self.card_l):
            self.model.addConstr(sum([self.P_B_giv_Yl[b, y, l] for b in range(self.card_B)]) == 1, name=f'sum P(b|{y,l}) = 1')


        self.model.addConstr(sum([self.P_l[l] for l in range(self.card_l)]) == 1, "sum_P_l = 1")

    def is_compatible(self, P_AB_giv_XY): # P_AB_giv_XY array
        self.initialize_model()

        # structural equation
        for a, b, x, y in np.ndindex(self.card_A, self.card_B, self.card_X, self.card_Y):
            self.model.addConstr(P_AB_giv_XY[a, b, x, y] == sum([self.P_A_giv_Xl[a, x, l] * self.prod[b, l, y] for l in range(self.card_l)]), f"P(A,B|X,Y) = sum_l P(A|X,l)*P(B|Y,l)*P(l)")

        self.model.update()
        self.model.optimize()

        if self.model.status == 2: # GRB.OPTIMAL
            return True
        else:
            self.model.computeIIS()
            return False
    
    def __str__(self):
        return self.P_AB_giv_XY
        

## testing:
dist = np.random.rand(2,2,2,2)  # rand dist
m = BellCompatible(2,2,2,2, card_l = 3, verbose=1)
model_feasibility = m.is_compatible(dist)
print("Is this compatible w/ Bell DAG?", model_feasibility)
print(m)