## 0.4.0 (2024-12-06)

### Feat

- adds workflow for pretraining a super-network (#173)
- allows to pass fabric (#162)

### Fix

- fix dependency versions to avoid breaking CI (#195)
- colab tutorial notebook and typos (#188)
- addition and standardization of docstrings (#176)
- Remove rope_cache from max_seq_len setter (#181)
- Revert "ci: reusing unit-test.yml in release.yml" (#187)
- release workflow clash with branch protection rules (#184)
- Revert "fix: reworking `release.yml` to avoid clashing with branch protection rules" (#182)
- reworking `release.yml` to avoid clashing with branch protection rules (#180)
- set input variables as required positional arguments (#172)
- update tokenizers in pyproject.toml (#168)
- type hints and docstring for `mkdocs build --clean --strict` (#167)
- forcing deepspeed to use CPU for profiling FLOPS (#154)
- deprecate flexible mlp heads (#160)

