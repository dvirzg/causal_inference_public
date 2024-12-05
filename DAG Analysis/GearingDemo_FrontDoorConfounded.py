"""
The DAG in question has Z-->A-->M-->Y with a common cause on {A,Y} and on {M,Y}.
We are using a gearing such that U_AY is tasked with determining A from Z
and U_MY is tasked with determining M from A and Y from {U_AY, M}.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

(cardZ, cardA, cardM, cardY) = (2,2,2,2)


A_from_Z_strategies = np.vstack(tuple(np.ndindex(tuple(np.broadcast_to(cardA, cardZ)))))
cardUAY = len(A_from_Z_strategies)

M_from_A_strategies = np.vstack(tuple(np.ndindex(tuple(np.broadcast_to(cardM, cardA)))))

Y_from_M_and_UAY_strategies = np.vstack(tuple(np.ndindex(tuple(np.broadcast_to(cardY, cardM*cardUAY)))))
Y_from_M_and_UAY_strategies = Y_from_M_and_UAY_strategies.reshape((-1, cardM, cardUAY))

shapeUMY = (len(M_from_A_strategies), len(Y_from_M_and_UAY_strategies))

env = gp.Env(empty=True)
env.setParam('LogToConsole', False) # To supress output
env.start()
m = gp.Model("qcp", env=env)
m.params.NonConvex = 2  # Using quadratic equality constraints.
m.params.LogFile = "solver_output.txt"
m.setParam('OptimalityTol', 0.01)


W_UAY = m.addMVar(cardUAY, lb=0, name="u_AY")
m.addConstr(W_UAY.sum()==1)
W_UMY = m.addMVar(shapeUMY, lb=0, name="u_MY")
m.addConstr(W_UMY.sum() == 1)
observable_internal = gp.MQuadExpr.zeros((cardZ, cardA, cardM, cardY))
do_internal = gp.MQuadExpr.zeros((cardA, cardY))

m.update()

for A_from_Z_idx, A_from_Z_strategy in enumerate(A_from_Z_strategies):
    w_UAY = W_UAY[A_from_Z_idx].item()
    UAY_val = A_from_Z_idx
    for M_from_A_idx, M_from_A_strategy in enumerate(M_from_A_strategies):
        for Y_from_M_and_UAY_idx, Y_from_M_and_UAY_strategy in enumerate(Y_from_M_and_UAY_strategies):
            w_UMY = W_UMY[M_from_A_idx, Y_from_M_and_UAY_idx].item()
            weight = w_UAY * w_UMY
            for z in range(cardZ):
                a_obs = A_from_Z_strategy[z]
                m_obs = M_from_A_strategy[a_obs]
                y_obs = Y_from_M_and_UAY_strategy[m_obs, UAY_val]
                observable_internal[z, a_obs, m_obs, y_obs] += weight
            for a_sharp in range(cardA):
                m_do = M_from_A_strategy[a_sharp]
                y_do = Y_from_M_and_UAY_strategy[m_do, UAY_val]
                do_internal[a_sharp, y_do] += weight



print(observable_internal)
#
# m.addConstr(P_obs == observable_internal)
# objective = do_internal[0, 1]
# m.setObjective(objective, sense=GRB.MAXIMIZE)
# m.optimize()
