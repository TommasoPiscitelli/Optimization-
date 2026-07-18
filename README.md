# LAD-SFSOD: Feature Selection and Outlier Detection under l1 norm

This project implements mixed-integer linear programming formulations for the simultaneous feature selection and outlier detection problem under the least absolute deviation criterion.

The project is based on the paper:

M. Barbato, A. Ceselli, "Mathematical programming for simultaneous feature selection and outlier detection under l1 norm", European Journal of Operational Research, 2024.

## Implemented models

At the current stage, the project implements the disjunctive-based MILP formulation, referred to as formulation (D) in the paper.

## Requirements

The project uses Python and Gurobi through the `gurobipy` interface.

## Files

- `test.py`: runs the implemented formulation on a small synthetic instance.
- `scalability.py`: performs a simple scalability analysis over synthetic instances of increasing size.
- `src/lad_sfsod/data.py`: synthetic instance generation.
- `src/lad_sfsod/model_d.py`: implementation of formulation (D).

## How to run

```bash
python test.py
python scalability.py