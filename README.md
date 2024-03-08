# Optimization of Demand Adaptive Systems at the Operational Level
## Python code used for this thesis.



### instance_generation.py
This is the main file. Here, we create the instances and run the different solution approaches for Chapter 5, Computational analysis.

### gurobi_model.py
In this file, we generate the Gurobi models for the arc-based MILP model formulation (Section 3.1) and path-based MILP model formulation (Section 3.2).

### grasp_heuristic.py
This file contains the code for the metaheuristic-based solution approach described in Sections 4.1 - 4.3.

### heuristic_path.py
In this file, we make the necessary changes for the second version that uses the path-based model (Section 4.4).

### case_study.py
This includes all relevant code for Chapter 6, Case study.


#### /data
This folder is empty, here we store our generated instances from instance_generation.py.

#### /data_case
This folder includes the generated pickle files that we generate and use in case_study.py.

