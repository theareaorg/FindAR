#!/bin/bash

python storage/run_parallel.py --startyear 2017 --updateNumber 4
python storage/preprocess.py --original_data_path storage/data/existing_dataset.csv --output_path storage/data
python storage/process.py --data_directory storage/data --output_folder_name storage/output
