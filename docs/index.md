# LDSFL-Meander documentation

This index groups the repository documentation by purpose. Start with the README
for installation and the bundled example, then use the guides below for detailed
configuration, validation, and release preparation.

## User guides

- [Short user manual](../USER_MANUAL.md)
- [CLI usage guide](cli_usage.md)
- [Timestep, erosion-rate, and iteration notes](timestep_notes.md)
- [macOS GUI and plotting notes](macos_gui_notes.md)
- [Full user manual PDF](LDSFL_Meander_user_manual.pdf)
- [Full user manual LaTeX source](LDSFL_Meander_user_manual.tex)

## Scientific interpretation and validation

- [Theory-to-code mapping](theory_code_mapping.md)
- [Validation strategy and known limitations](validation_strategy.md)
- [Validation notes](validation_notes.md)
- [Reproducibility checklist](reproducibility_checklist.md)
- [Run manifest template](run_manifest_template.md)

## Maintenance and release preparation

- [Project health checklist](project_health_checklist.md)
- [v0.6.5 release-notes draft](release_notes_v0_6_5.md)
- [Portfolio summary](portfolio_summary.md)

## Current interface reminders

- The reference backend is NumPy.
- The default flow boundary condition is free/open.
- The historical bank-erodibility default is `1.0e-8`.
- `--erosion-rate` changes simulated-time scaling much more than geometry per
  solver step because the timestep adapts inversely to migration speed.
- Resonance and sinuosity diagnostics are interpretation aids, not substitutes
  for external physical validation.
- Generated `Output/` directories should normally remain outside version control.
