## Overview

This Python script allows you to download research articles from Engineering Village using the Elsevier API. Engineering Village is a comprehensive engineering research database that provides access to a vast collection of scholarly articles, conference papers, and other engineering-related resources. The script is run in a docker container.

## Requirements

Before using this script, make sure you have the following prerequisites installed:

- An engineering village institutional token and an api token, save these into the file vars.env
- Docker for desktop

## Usage

1. Clone or download this repository to your local machine.

2. Obtain an Elsevier API key:

   - Contact Elsevier to get an appropriate API key with access to Engineering village

3. Configure the environment variables:

   - Open the `vars.env` file and replace `ELSAPI` with your Elsevier API key and `ELSINST` with your insitutional token.

4. Start the docker container:

   - Open your terminal or command prompt and navigate to the project directory.
   - Start the docker container by running `docker compose up`.

## How it works

There are two modules in the library one module is used for downloading the files and the second is used for running analysis on the resulting dataset.
The docker container runs the script `runall.sh` at startup which calls three python routines

- `run_parallel.py` - is used to download new articles from engineering village and will use all available cpus to download the articles in parallel.
- `preprocess.py` - merges the downloaded articles with the existing dataset
- `process.py` - applies the analysis routine to the merged dataset

Static files used during the computation are stored on the google bucket `gs://area-findar/data`.
These are downloaded automatically when run using the run.sh script, but will need to be manually downloaded if running the code locally.

### Downloader

The downloader searches the Engineering Village database for the phrase "augmented reality" for any relevant conference articles or journal articles. The user can refine the search by specifying the start year, which will return all results following that year, the end year which will return all. Alternatively, the parameter `update_number=N` can be used for searching for any new articles published in the previous `N` weeks, however the update number must be less than 4.

Once the downloader has found the articles it will download all of the available meta data for the articles and save these into a csv file.

The downloader can be run by calling the run_parallel.py function with arguments to specify the mode for example:
`run_parallel.py --startyear 2017 --endyear -2018` would download all of the articles between 2017 and 2018
`run_parallel.py --updateNumber 4` would download the new articles in the last 4 weeks.

The results are saved into a csv file `storage\dataset.csv` however this can be specified using the `--ouput_path` argument.

A helper script `preprocess2.py` can be used to merge the existing dataset with the newly downloaded dataset and removing any duplicates (a duplicate will have the same DOI).

### Analysis

The analysis module is an adaption of (https://github.com/theareaorg/AREA-Research-Agenda/blob/main/FindAR/Code/AR_papers_analysis.ipynb). The module applies a natural language processing workflow to the dataset.

The following steps are used by the NLP workflow:

1.  Merge the newly downloaded dataset with the existing dataset
2.  Preprocess (remove stop words, punctuation and apply stemming)
3.  Extract all low level, mid level and high level terms from the dataset
4.  Remove/replace terms with a dictionary stored in `replacements-new.csv`
5.  Merge all tags together
6.  Caclulate frequency of the terms for the dataset

The outputs that are used in the findar_search tool are stored in the json folder:

- `category.json` - contains

- `data.json` - export of the pandas dataframe containing all of the articles

- `terms.json` -

- `topics.json` - contains the database of mid level terms and the low level terms that are assocaited with these

The analysis can be run using the `process.py` file.

## Deploying on the cloud

The workflow can be deployed onto a cloud services e.g. google cloud, aws, azure. To do this you will need to make a linux compute instance using the cloud environment. Then clone this git repository onto the cloud server and follow the local instructions

## Updating term mapping

To update high/mid/low level mapping change the `./Data/terms-bucketing.csv` file and restart and run the docker container.

### Output files

There are three directories of output files

1. figures

   - low level frequency bar charts are stored in low_page.pdf
   - cross plot of abstract word count and mean similarity
   - high level term pairs scatter plot weighted by the frequency of the term

2. json

   -

3. text

   -
