# Model Scope and Validation Notes

## Purpose

LDSFL-Meander is a reduced morphodynamic model for exploratory studies of meander evolution. It is intended for reproducible reduced-model experiments, teaching, sensitivity testing and method comparison.

## Intended use

The model is most appropriate for:

- wide, mildly curved, long river bends
- reduced-order morphodynamic studies
- exploratory simulations
- comparison of parameter effects
- teaching and reproducible research examples

## Not intended for

The model should not be presented as:

- a full 2D or 3D hydrodynamic solver
- a sharp-bend flow-separation model
- a calibrated engineering design tool
- a flood-risk model
- a site-specific prediction model without additional validation

## Main assumptions

The public version uses simplified/reduced assumptions to make meander evolution computationally accessible and reproducible. Users should review the notation, input parameters and manual before interpreting results.

## Reproducibility checks

The repository includes:

- a command-line smoke test
- a GUI/configuration smoke test
- pytest-based test execution
- GitHub Actions continuous integration
- citation metadata and archived DOI release

## Validation status

The current repository version is intended as a clean public research-software release. It includes smoke tests and reproducible examples, but users should perform independent validation before using the model for scientific conclusions or site-specific applications.

## Recommended citation

Use the citation metadata in `CITATION.cff` and the Zenodo DOI listed in the README.
