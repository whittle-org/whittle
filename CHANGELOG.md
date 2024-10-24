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
