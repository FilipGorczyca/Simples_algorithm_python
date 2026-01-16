# Simples_algorithm_python

The simplex algorithm is used to find the optimal solution to a linear programming problem—maximizing profit or minimizing cost subject to linear constraints (e.g., limited resources). It works iteratively by moving between corner points (vertices) of the feasible region, improving the objective value until it reaches the optimum.

Python/NumPy implementation of the Simplex algorithm (Big‑M) for linear programming with mixed constraints (<=, >=, =), returning the optimal solution and objective value
This repository contains a Python (NumPy) implementation of the Simplex method for solving linear programming problems. The solver supports constraints of type <=, >= and = by introducing slack/surplus/artificial variables using the Big‑M approach. It returns the optimal decision vector and the objective function value and includes an example problem to reproduce results

Max and min problems (min handled via sign conversion).
​
Support for <=, >=, = constraints (slack/surplus/artificial variables; Big‑M penalty).
​
Outputs: solution vector + objective value (optionally iteration logs if enabled).
