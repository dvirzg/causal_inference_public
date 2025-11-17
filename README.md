# causal_inference_public

Some of the non-proprietary work and exploration I did while at Perimeter

## Overview

This repository contains research on causal inference using nonconvex quadratic programming (Gurobi) to estimate causal effects, test graphical model compatibility, and perform counterfactual reasoning when hidden confounders exist.

## Main Research Areas (Favourites/)

### 1. Unpacking DAGs (`Favourites/Unpacking DAGs/`)
Transforms causal DAGs into counterfactual graphs to estimate **causal intervention effects** with hidden confounders. Nodes are "unpacked" into counterfactual versions (e.g., $A \rightarrow \{A_0, A_1\}$), using quadratic separability constraints to compute bounds on causal effects $P(Y | \text{do}(A=a))$.

**Key files:**
- `unpacking_front_door_confounded.ipynb` - Main implementation with full documentation
- `GearingDemo_FrontDoorConfounded.py/ipynb` - Alternative "gearing" approach
- `interuption_dag_dvirs*.ipynb` - Interruption DAG scenarios

### 2. Compatibility Testing (`Favourites/Compatibility/`)
Tests if observed probability distributions are **compatible** with specific causal graph structures (Bell DAG). Determines whether correlations $P(A,B|X,Y)$ can be explained by a classical hidden common cause using the structural equation: $P(A,B|X,Y) = \sum_\lambda P(A|X,\lambda) \cdot P(B|Y,\lambda) \cdot P(\lambda)$.

**Key files:**
- `bell_compatibility_class.py` - Reusable class for Bell compatibility testing
- `Bell Compatibility.ipynb` - Full implementation
- `Latent Var Min Card/` - Find minimum cardinality of hidden variable needed

### 3. Inflation (`Favourites/Inflation/`)
Creates copies of a causal DAG to derive **new testable constraints** on probability distributions. Used for compatibility testing and causal effect estimation by relating inflated graphs back to the original via constraints.

**Key files:**
- `ring_inflation_of_triangle_compatibility_matrix_form.ipynb` - Triangle DAG ring inflation
- `triangle_inflation_do_cond_diff.ipynb` - Estimating intervention effects using inflation
- `cut_inflation_of_triangle_compatibility_test.ipynb` - Alternative inflation schemes

### 4. Distribution Generation (`Favourites/Dist_Generation from DAG/`)
Generates probability distributions **compatible** with specific DAG structures using deterministic strategies. Uses the 16 deterministic strategies for binary Bell scenarios as extremal points to create convex combinations.

**Key files:**
- `generating_compatible_dists_w_Bell.ipynb` - Generate Bell-compatible distributions
- `generate_convex_points_for_funcs.ipynb` - Generate extremal points (deterministic strategies)
- `v_checker.py` - Parameterized compatibility checker
