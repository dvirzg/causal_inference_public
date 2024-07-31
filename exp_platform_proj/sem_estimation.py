import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np

class sem_estimator:
    def __init__(self, graph, hiddens_lst, cards_dict): 
        self.graph = nx.DiGraph()
        self.graph.add_edges_from((parent, child) for parent, child in graph)
        self.verbose = 0
        self.hiddens_lst = hiddens_lst
        self.cards = cards_dict
        self.prob_vars = {}
        
        # landing
        print("This library is used to estimate the SEM of a DAG with hidden variables.\nSpecify the DAG structure, the observable and hidden nodes and their cardinalities, then specify your loss function.\n")

    def get_mvar_details(self, var):
        parents = list(self.graph.predecessors(var))
        # returns tuple(card_var, *card_parents(var)), 
        return (self.cards[var], *tuple(self.cards[parent] for parent in parents)), "_".join(parents)

    def initialize_model(self):
        self.model = gp.Model("DAG-Dist Operations")
        self.model.reset()
        self.model.setParam('OutputFlag', self.verbose)
        self.model.params.NonConvex = 2

        ## variables
        for var in list(self.graph.nodes):
            shape, var_name = self.get_mvar_details(var)
            if not var_name:
                self.prob_vars[f'P_{var}'] = self.model.addMVar(shape=shape, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'P_{var}')
            else:
                self.prob_vars[f'P_{var}_giv_{var_name}'] = self.model.addMVar(shape=shape, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'P_{var}_giv_{var_name}')
        self.model.update()

    def markv_decomp_prod(self, coords, hidden_val):
        ...


    def get_joint(self):
        self.observables_lst = [var for var in self.cards.keys() if var not in self.hiddens_lst]
        self.childless = [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]
        self.given = [node for node in self.graph.nodes if node not in self.childless and node not in self.hiddens_lst]
        self.target_joint_prob = "".join(self.childless) + "_giv_" + "".join(self.given)
        self.observables_cards = tuple(self.cards[var] for var in self.observables_lst)
        self.hidden_cards = tuple(self.cards[var] for var in self.hiddens_lst)
        self.observable_combinations = itertools.product(*[range(card) for card in self.observables_cards])

        self.prob_vars[f'P_{self.target_joint_prob}'] = self.model.addMVar(shape= self.observables_cards, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'P_{self.target_joint_prob}')
        self.model.update()
        # ...

        # print(self.prob_vars)
        print("Model constraints:", '\n...\n')


    def specify_loss(self, loss_func):
        self.loss_func = loss_func
        print("Loss function defined as", loss_func)

    def outsourcing_model(self):
        # self.model.optimize()
        print("Model has been optimized, weights saved.")
        # print("Optimal value:", self.model.objVal)
        # print("Optimal solution:")
        # for v in self.model.getVars():
        #     print(v.varName, v.x)
        # print("Model has been optimized.")
        # return self.model.objVal
        print("Done.")
    
    def fit_dist(self, df):
        self.initialize_model()
        self.get_joint()
        print(f"Fitting Distribution based on {self.loss_func}")
        self.outsourcing_model()
        # self.model.optimize()
