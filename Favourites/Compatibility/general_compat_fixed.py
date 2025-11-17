import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np

class DAG_Dist_Compatibility:
    def __init__(self, graph, hiddens_lst, cards_dict, verbose=0):
        """
        Initialize compatibility tester for arbitrary DAG.

        Args:
            graph: List of [parent, child] edges
            hiddens_lst: List of hidden variable names
            cards_dict: Dict mapping variable names to cardinalities
            verbose: Gurobi output verbosity (0=silent, 1=verbose)
        """
        self.graph = nx.DiGraph()
        self.graph.add_edges_from((parent, child) for parent, child in graph)
        self.verbose = verbose
        self.hiddens_lst = hiddens_lst
        self.cards = cards_dict
        self.prob_vars = {}

        # Ensure consistent ordering of variables
        self.observables_lst = [var for var in sorted(self.graph.nodes) if var not in self.hiddens_lst]
        self.observables_dict = {var: idx for idx, var in enumerate(self.observables_lst)}

    def get_mvar_details(self, var):
        """
        Get variable shape and parent information.

        Returns:
            shape: tuple (card_var, card_parent1, card_parent2, ...)
            parent_names: list of parent variable names
        """
        parents = list(self.graph.predecessors(var))
        shape = (self.cards[var], *tuple(self.cards[parent] for parent in parents))
        return shape, parents

    def initialize_model(self):
        """Create Gurobi model and add probability variables for each node."""
        self.model = gp.Model("DAG-Dist Compatibility")
        self.model.reset()
        self.model.setParam('OutputFlag', self.verbose)
        self.model.params.NonConvex = 2

        # Create probability variables P(var | parents)
        for var in self.graph.nodes:
            shape, parents = self.get_mvar_details(var)
            if not parents:
                # Marginal probability P(var)
                self.prob_vars[f'P_{var}'] = self.model.addMVar(
                    shape=shape, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'P_{var}'
                )
            else:
                # Conditional probability P(var | parents)
                parent_str = "_".join(parents)
                self.prob_vars[f'P_{var}_giv_{parent_str}'] = self.model.addMVar(
                    shape=shape, vtype=GRB.CONTINUOUS, lb=0, ub=1,
                    name=f'P_{var}_giv_{parent_str}'
                )

        # Add normalization constraints
        self._add_normalization_constraints()

        self.model.update()

    def _add_normalization_constraints(self):
        """Add constraints that probabilities sum to 1."""
        for var in self.graph.nodes:
            shape, parents = self.get_mvar_details(var)
            var_card = self.cards[var]

            if not parents:
                # P(var) must sum to 1
                prob_var = self.prob_vars[f'P_{var}']
                self.model.addConstr(
                    sum(prob_var[i] for i in range(var_card)) == 1,
                    name=f'normalize_P_{var}'
                )
            else:
                # For each parent configuration, P(var | parents) must sum to 1
                parent_str = "_".join(parents)
                prob_var = self.prob_vars[f'P_{var}_giv_{parent_str}']
                parent_cards = [self.cards[p] for p in parents]

                for parent_config in itertools.product(*[range(c) for c in parent_cards]):
                    self.model.addConstr(
                        sum(prob_var[i, *parent_config] for i in range(var_card)) == 1,
                        name=f'normalize_P_{var}_giv_{parent_config}'
                    )

    def _get_variable_index_in_coords(self, var, coords):
        """
        Get the index value for variable 'var' from coords tuple.

        Args:
            var: Variable name
            coords: Tuple of values for observable variables in canonical order

        Returns:
            Index value for this variable, or None if not in coords
        """
        if var in self.observables_dict:
            return coords[self.observables_dict[var]]
        return None

    def markv_decomp_prod(self, coords, hidden_vals):
        """
        Compute the Markov factorization product for given observable coords and hidden values.

        P(observables, hidden) = ∏ P(node | parents(node))

        Args:
            coords: Tuple of observable variable values (in canonical order)
            hidden_vals: Tuple of hidden variable values

        Returns:
            Gurobi expression representing the product
        """
        hidden_dict = {h: hidden_vals[i] for i, h in enumerate(self.hiddens_lst)}

        terms = []

        # Iterate through all nodes in topological order
        for var in nx.topological_sort(self.graph):
            shape, parents = self.get_mvar_details(var)

            if not parents:
                # No parents: P(var)
                var_idx = self._get_variable_index_in_coords(var, coords)
                if var_idx is None:  # var is hidden
                    var_idx = hidden_dict[var]

                terms.append(self.prob_vars[f'P_{var}'][var_idx])
            else:
                # Has parents: P(var | parents)
                parent_str = "_".join(parents)

                # Get index for var itself
                var_idx = self._get_variable_index_in_coords(var, coords)
                if var_idx is None:  # var is hidden
                    var_idx = hidden_dict[var]

                # Get indices for parents
                parent_indices = []
                for parent in parents:
                    p_idx = self._get_variable_index_in_coords(parent, coords)
                    if p_idx is None:  # parent is hidden
                        p_idx = hidden_dict[parent]
                    parent_indices.append(p_idx)

                # Access P(var | parents)[var_idx, parent_idx1, parent_idx2, ...]
                full_index = (var_idx, *parent_indices)
                terms.append(self.prob_vars[f'P_{var}_giv_{parent_str}'][full_index])

        # Multiply all terms using auxiliary variables to avoid higher-order products
        return self._multiply_terms(terms)

    def _multiply_terms(self, terms):
        """
        Multiply a list of Gurobi variables/expressions using auxiliary variables.

        Since Gurobi can only handle quadratic constraints, we need to multiply
        pairwise and introduce auxiliary variables.
        """
        if len(terms) == 0:
            return 1
        if len(terms) == 1:
            return terms[0]

        # Reduce pairwise
        current_terms = terms[:]
        aux_counter = 0

        while len(current_terms) > 1:
            new_terms = []
            for i in range(0, len(current_terms), 2):
                if i + 1 < len(current_terms):
                    # Create auxiliary variable for product
                    aux_var = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, ub=1,
                        name=f'aux_prod_{aux_counter}'
                    )
                    self.model.addConstr(
                        aux_var == current_terms[i] * current_terms[i + 1],
                        name=f'aux_constr_{aux_counter}'
                    )
                    new_terms.append(aux_var)
                    aux_counter += 1
                else:
                    # Odd one out, carry forward
                    new_terms.append(current_terms[i])
            current_terms = new_terms

        return current_terms[0]

    def is_compatible(self, P_obs):
        """
        Test if observed distribution P_obs is compatible with DAG structure.

        Args:
            P_obs: Numpy array of observed probabilities
                   Shape should match (card_obs1, card_obs2, ..., card_obsN)
                   where observables are in canonical sorted order

        Returns:
            True if compatible, False otherwise
        """
        self.initialize_model()

        # Add structural equation constraints
        # P(observables) = Σ_hidden ∏ P(node | parents)

        obs_cards = [self.cards[var] for var in self.observables_lst]
        hidden_cards = [self.cards[var] for var in self.hiddens_lst]

        for obs_config in itertools.product(*[range(c) for c in obs_cards]):
            # Sum over all hidden configurations
            rhs_terms = []
            for hidden_config in itertools.product(*[range(c) for c in hidden_cards]):
                markov_product = self.markv_decomp_prod(obs_config, hidden_config)
                rhs_terms.append(markov_product)

            # LHS: observed probability
            lhs = P_obs[obs_config]

            # Constraint: P(obs_config) = Σ_hidden ∏ P(node | parents)
            self.model.addConstr(
                lhs == sum(rhs_terms),
                name=f'SEM_{obs_config}'
            )

        self.model.update()
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print(f"✓ Distribution is compatible with DAG")
            return True
        else:
            print(f"✗ Distribution is NOT compatible with DAG")
            if self.verbose:
                self.model.computeIIS()
                print("Irreducible Infeasible Subsystem computed")
            return False

    def get_fitted_distributions(self):
        """
        After successful optimization, extract the fitted P(node | parents) distributions.

        Returns:
            Dictionary mapping variable names to their conditional distributions
        """
        if self.model.status != GRB.OPTIMAL:
            raise ValueError("Model not optimized or infeasible")

        fitted = {}
        for var in self.graph.nodes:
            shape, parents = self.get_mvar_details(var)
            if not parents:
                fitted[f'P_{var}'] = self.prob_vars[f'P_{var}'].X
            else:
                parent_str = "_".join(parents)
                fitted[f'P_{var}_giv_{parent_str}'] = self.prob_vars[f'P_{var}_giv_{parent_str}'].X

        return fitted


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing General DAG Compatibility")
    print("=" * 80)

    # Test 1: Bell DAG (X→A←λ→B←Y)
    print("\n--- Test 1: Bell DAG ---")
    bell_edges = [["X", "A"], ["l", "A"], ["l", "B"], ["Y", "B"]]
    bell_cards = {"A": 2, "B": 2, "X": 2, "Y": 2, "l": 3}
    bell_hiddens = ["l"]

    bell_model = DAG_Dist_Compatibility(bell_edges, bell_hiddens, bell_cards, verbose=0)

    # Test with random distribution (likely incompatible)
    P_obs_random = np.random.rand(2, 2, 2, 2)  # P(A, B, X, Y)
    P_obs_random /= P_obs_random.sum()  # Normalize

    print(f"Testing random distribution...")
    is_compat = bell_model.is_compatible(P_obs_random)

    # Test 2: Simple DAG (Z→A←λ→B)
    print("\n--- Test 2: Simple DAG (Z→A←λ→B) ---")
    simple_edges = [["Z", "A"], ["l", "A"], ["l", "B"]]
    simple_cards = {"A": 2, "B": 2, "Z": 2, "l": 3}
    simple_hiddens = ["l"]

    simple_model = DAG_Dist_Compatibility(simple_edges, simple_hiddens, simple_cards, verbose=0)

    # Create a distribution that SHOULD be compatible
    # (by construction: generate from a specific latent model)
    P_obs_test = np.zeros((2, 2, 2))  # P(A, B, Z) in alphabetical order

    # Generate compatible distribution
    P_l = np.array([0.3, 0.4, 0.3])  # P(l)
    P_Z = np.array([0.6, 0.4])  # P(Z)
    P_A_giv_Zl = np.random.rand(2, 2, 3)  # P(A | Z, l)
    P_A_giv_Zl /= P_A_giv_Zl.sum(axis=0, keepdims=True)  # Normalize
    P_B_giv_l = np.random.rand(2, 3)  # P(B | l)
    P_B_giv_l /= P_B_giv_l.sum(axis=0, keepdims=True)  # Normalize

    for a, b, z in itertools.product(range(2), range(2), range(2)):
        P_obs_test[a, b, z] = sum(
            P_A_giv_Zl[a, z, l] * P_B_giv_l[b, l] * P_Z[z] * P_l[l]
            for l in range(3)
        )

    print(f"Testing constructed compatible distribution...")
    is_compat = simple_model.is_compatible(P_obs_test)

    if is_compat:
        fitted = simple_model.get_fitted_distributions()
        print("\nFitted distributions:")
        for key, val in fitted.items():
            print(f"  {key}: shape {val.shape}")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
