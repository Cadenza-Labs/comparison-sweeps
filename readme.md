# How to use

```bash
# my approach is to create a sweep file for
# each dataset-model pair (9 x 5 = 35)
# each uses 1 gpu, that should make it more distrbutable (this helps for running on cais, and maybe spar too)

# each config file then gets converted into its corresponding sweep file and saved, no magic here
python config_gen.py
python configs_to_scripts.py
```