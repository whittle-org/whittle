## 0.5.1 (2025-04-22)

### Fix

- update index in docs (#280)
- use current pip version of syne-tune (#290)

## 0.5.0 (2025-04-16)

### Feat

- add workflow finetune (#275)
- lora finetuning (#265)
- Add new distill losses (#266)
- Checkpointing and convert to litgpt (#268)
- add distillation workflow (#261)
- litgpt-style CLI (#246)
- remove duplicated lora QKV linear layer (#260)
- adds pruning workflow (#241)
- search results and evaluation workflow (#249)
- add lora supernet (#245)
- make deepspeed install optional (#248)
- add pypi version to readme (#234)
- add workflow for multi-objective search (#179)
- addition of structural pruning methods (#192)

### Fix

- optimize lora_qkv_linear (#274)
- link to distillation workflow in readme (#269)
- off-by-one error in bin sampling (#267)
- lora indexing (#259)
- use `litgpt.data.Alpaca` instead of C4 in pruning workflow (#257)
- pass dataloader in data loading function (#240)
- gqa and mqa indexing (#244)
- update contributor list (#250)
- ruff linting rules (#253)
- formatting (#252)
- add link for search workflow to README (#232)
- use random init for unit test (#227)
- updating `ruff` config to respect `target-version` for linting (#224)
- Reset the rope cache in reset_super_network (#220)
- update doc for puning (#218)
- renamed folder from sine_curves/ to sinc/ (#219)
- Enable to extract the current active subnet without passing configs/dicts. (#213)

### Refactor

- Use Lightning instead of Deepspeed to measure flops (#281)

## 0.4.1 (2024-12-12)

### Fix

- tagging
- bump

## 0.4.0 (2024-12-12)

### Feat

- adds workflow for pretraining a super-network (#173)
- allows to pass fabric (#162)

### Fix

- **release.yml**: artifact upload for changelog (#211)
- **release.yml**: make bump dependent on test-code (#209)
- set cos sin in max_seq_len (#203)
- update readme (#193)
- release workflow (#202)
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

## 0.3.0 (2024-10-24)

### Feat

- add support for LLamaMLP in extract_sub_network (#147)
- adding flops and macs profiling for subnets (#145)
- add script to profile latency (#141)
- modify rope for llama-3 and support llama-3.2 (#131)
- add gpt tutorial notebook utils (#122)
- add installation instruction to documentation (#121)

### Fix

- refactor names of metric (#152)
- Extract weights for norm layers, test with random initialization. (#151)
- handle device in GPT model properly (#143)
- rename call function (#144)
- delete supernet_configs directory (#140)
- deprecate  sample_random_indices (#133)
- support params, mag when sharing layer norm in phi-2 (#127)
- reset random layers in reset_super_network (#126)
- support GQA param count (#124)
- update readme (#111)

## 0.2.0 (2024-09-08)

### Feat

- litgpt update (#95)

### Fix

- adding cz config (#119)
- removing version parsing in whittle/__init__.py (#118)
- renaming whittle/version to whittle/__version__.py (#117)
- commitizen configuration (#115)
- remove deprecated module (#105)
- delete old code (#104)
- allow to pass other loss function to training strategies (#101)
- set random state properly in sampler (#103)
