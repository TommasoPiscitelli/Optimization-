# LAD-SFSOD: Feature Selection and Outlier Detection under l1 norm

This project implements mixed-integer linear programming formulations for the simultaneous feature selection and outlier detection problem under the least absolute deviation criterion.

The project is based on the paper:

M. Barbato, A. Ceselli, "Mathematical programming for simultaneous feature selection and outlier detection under l1 norm", European Journal of Operational Research, 2024.

## Implemented models

At the current stage, the project implements two MILP formulations from the paper:

- **Formulation (D)**: the disjunctive-based formulation. This is the main formulation used in the project and is expected to provide a stronger linear relaxation.

- **Formulation (L)**: an alternative linear formulation based on auxiliary residual-correction variables. This model is included mainly for comparison with formulation (D), both in terms of solution quality and computational behavior.

Both formulations are tested on the synthetic instances provided by the original authors, with the aim of comparing their performance under different values of the feature-selection budget and the outlier-detection budget.
