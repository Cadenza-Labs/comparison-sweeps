#! /bin/bash
rm -rf scripts configs
python config_gen.py
python configs_to_scripts.py