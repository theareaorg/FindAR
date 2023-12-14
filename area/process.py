import pandas
import subprocess, os
import seaborn as sns
import ast
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import re
from sklearn import preprocessing as pre
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import json
from pathlib import Path

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
from collections import OrderedDict
from operator import itemgetter
import itertools
from collections import Counter
import json
import os
from scipy.stats import norm


def preprocess_alias(alias):
    """ """
    index_to_drop = []
    index_to_drop.append(alias.loc[alias["LowLevel"] == "multi-robot systems"].index[1])
    index_to_drop.append(alias.loc[alias["LowLevel"] == "systematic errors"].index[1])
    index_to_drop.append(
        alias.loc[alias["LowLevel"] == "ultra-wideband (uwb)"].index[1]
    )
    index_to_drop.append(alias.loc[alias["LowLevel"] == "magnetometers"].index[1])
    alias = alias.drop(index_to_drop)
    alias = alias.reset_index(drop=True)
    return alias


def process_text(text_list, stop_words=None):
    print("Processing text...")
    new_text_list = []
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    for i in range(0, len(text_list)):
        # Remove punctuations
        text = re.sub("[^a-zA-Z0-9]", " ", text_list[i])

        # Convert to lowercase
        text = text.lower()

        # remove tags
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

        # remove special characters and digits
        # text=re.sub("(\\d|\\W)+"," ",text)

        ##Convert to list from string
        text = text.split()

        ##Stemming
        # ps = PorterStemmer()
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in stop_words]
        text = " ".join(text)
        new_text_list.append(text)
    return new_text_list


def map_edges_to_alias(edges, alias):
    print("Mapping alias to edges")
    source_terms = edges.loc[:, "Source"].to_numpy()
    target_terms = edges.loc[:, "Target"].to_numpy()
    terms = alias.loc[:, "LowLevel"].to_numpy()
    i, j = np.where(source_terms[:, None] == terms[None, :])
    edges.loc[i, "midlevel-source"] = list(alias.loc[j, "MidLevel"])
    i, j = np.where(target_terms[:, None] == terms[None, :])
    edges.loc[i, "midlevel-target"] = list(alias.loc[j, "MidLevel"])
    term = alias["MidLevel"].unique().tolist()
    if "Label" not in edges:
        edges["Label"] = ""

    edges.loc[
        edges["midlevel-source"] == edges["midlevel-target"], "Label"
    ] = edges.loc[
        edges["midlevel-source"] == edges["midlevel-target"], "midlevel-source"
    ].to_list()

    source = edges["midlevel-source"].to_numpy()
    target = edges["midlevel-target"].to_numpy()
    term_np = np.array(term)

    i, j = np.where(source[:, None] == term_np[None, :])
    i2, j2 = np.where(target[:, None] == term_np[None, :])

    iall = np.hstack([i, i2])
    jall = np.hstack([j, j2])
    edge_mat = edges[term].to_numpy()
    edge_mat[:] = 0
    edge_mat[iall, jall] = 1
    edges[term] = edge_mat
    return edges


def codes_from_dataset(dataset, CLASS_CODE, CONTROLLED_CODE, UNCONTROLLED_CODE):
    print("Extracting codes from dataset")
    control_keywords = []
    uncontrol_keywords = []
    classification_code = []

    for i in range(0, len(dataset)):
        class_code = dataset[CLASS_CODE][i]
        res = isinstance(class_code, str)
        if res == False:
            dataset.loc[i, CLASS_CODE] = "other"
            class_code = dataset[CLASS_CODE][i]
        class_code = re.split(";", class_code)

        control_keys = dataset[CONTROLLED_CODE][i]
        res = isinstance(control_keys, str)
        if res == False:
            dataset.loc[i, CONTROLLED_CODE] = "other"
            control_keys = dataset.loc[i, CONTROLLED_CODE]
        # dataset['LowLevel'] = control_keys.lower()
        control_keys = re.split(";", control_keys)

        uncontrol_keys = dataset.loc[i, UNCONTROLLED_CODE]
        res = isinstance(uncontrol_keys, str)
        if res == False:
            dataset.loc[i, UNCONTROLLED_CODE] = "other"
            uncontrol_keys = dataset.loc[i, UNCONTROLLED_CODE]
        uncontrol_keys = re.split(";", uncontrol_keys)

        for j in range(0, len(control_keys)):
            control_keys[j] = control_keys[j].lower()

        for k in range(0, len(uncontrol_keys)):
            uncontrol_keys[k] = uncontrol_keys[k].lower()

        # for l in range(0,len(class_code)):
        # class_code[l] = class_code[l].lower()

        control_keywords.append(control_keys)
        uncontrol_keywords.append(uncontrol_keys)
        classification_code.append(class_code)
    return control_keywords, uncontrol_keywords, classification_code


def add_lowlevel(dataset, CONTROLLED_CODE, replacement_terms):
    print("Adding low level terms")
    dataset["LowLevel"] = dataset[CONTROLLED_CODE].str.lower()
    for i in range(0, len(dataset)):
        low = dataset.loc[i, "LowLevel"]
        terms = low.split(";")
        terms_to_delete = []
        for l in range(0, len(terms)):
            term = terms[l]
            for k in range(0, len(replacement_terms)):
                check_term = replacement_terms["FIND"][k]
                replace_term = replacement_terms["REPLACE"][k]
                if check_term == term:
                    if replacement_terms["REPLACE"][k] == "DELETE":
                        terms_to_delete.append(term)
                    if replacement_terms["REPLACE"][k] != "DELETE":
                        terms[l] = replace_term

        for m in range(0, len(terms_to_delete)):
            terms.remove(terms_to_delete[m])

        dataset.loc[i, "LowLvlList"] = json.dumps(terms)
        dataset.loc[i, "lowlvl"] = ";".join(terms)
        dataset.loc[i, "LowLevel"] = json.dumps(terms)

    dataset.loc[dataset["lowlvl"] == "", "lowlvl"] = "other"

    # print(counter)
    return dataset


def apply_term_bucketing(
    dataset: pandas.DataFrame, alias: pandas.DataFrame, inplace: bool = False
):
    """
    Search through the alias's and associate low level terms with high level terms
    find any new low level terms and associate with other

    Parameters
    ----------
    dataset : pd.DataFrame
        dataset containing articles
    alias : pd.DataFrame
        association between lowlevel,mid and high
    inplace : bool
        apply to dataset inplace or create new one
    Returns
    -------
    pd.Dataframe
        updated dataset with mid/high level terms
    """
    if inplace == False:
        dataset = dataset.copy()
    print("Adding high level terms")
    all_mids = []
    all_highs = []
    dataset["MidLevel"] = ""
    dataset["HighLevel"] = ""
    # get all of the unique low level terms from the dataset
    unique_lows = set(";".join(dataset["lowlvl"].to_list()).split(";"))
    # get all of the unique low level terms from the alias
    alias_lows = set(alias["LowLevel"].to_list())
    # get the difference between the two
    diff = list(unique_lows.difference(alias_lows))
    new_terms = pandas.DataFrame(diff, columns=["LowLevel"])
    new_terms["MidLevel"] = "other"
    new_terms["HighLevel"] = "other"
    # add the difference to the alias
    alias = pandas.concat([alias, new_terms], ignore_index=True)
    alias = alias.reset_index()
    for j in alias.index:
        dataset.loc[
            dataset["lowlvl"].str.contains(alias.loc[j, "LowLevel"], regex=False),
            "MidLevel",
        ] += (
            alias.loc[j, "MidLevel"] + ";"
        )
        dataset.loc[
            dataset["lowlvl"].str.contains(alias.loc[j, "LowLevel"], regex=False),
            "HighLevel",
        ] += (
            alias.loc[j, "HighLevel"] + ";"
        )

    # find all of the unique high level terms for each article
    all_highs = [
        ";".join(list(set([h for h in hlvl.split(";") if len(h) > 0])))
        for hlvl in dataset["HighLevel"].to_list()
    ]
    # find all of the unique mid level terms for each article
    all_mids = [
        ";".join(list(set([m for m in mlvl.split(";") if len(m) > 0])))
        for mlvl in dataset["MidLevel"].to_list()
    ]

    dataset["HighLevel"] = all_highs
    dataset["MidLevel"] = all_mids
    dataset["midlvl"] = dataset["MidLevel"]
    dataset["highlvl"] = dataset["HighLevel"]

    return dataset


def combined_tags(dataset, UNCONTROLLED_CODE, CLASS_CODE):
    print("Adding combined tags")
    combined_tags = []
    idea_tags = []
    abs_tags = []
    skip_index = []
    # df.drop(columns=['combinedTags'])
    for i in range(0, len(dataset)):
        if pandas.isna(dataset.loc[i, "lowlvl"]) == True:
            lowlevel_terms = ""
            skip_index.append(i)
        if pandas.isna(dataset.loc[i, "lowlvl"]) == False:
            lowlevel_terms = dataset.loc[i, "lowlvl"]

        if pandas.isna(dataset.loc[i, UNCONTROLLED_CODE]) == True:
            uncontrol_terms = ""
            skip_index.append(i)
        if pandas.isna(dataset.loc[i, UNCONTROLLED_CODE]) == False:
            uncontrol_terms = dataset.loc[i, UNCONTROLLED_CODE]

        if pandas.isna(dataset.loc[i, CLASS_CODE]) == True:
            class_terms = ""
            skip_index.append(i)
        if pandas.isna(dataset.loc[i, CLASS_CODE]) == False:
            class_terms = dataset.loc[i, CLASS_CODE]

        if pandas.isna(dataset.loc[i, "midlvl"]) == True:
            midlevel_terms = ""
            skip_index.append(i)
        if pandas.isna(dataset.loc[i, "midlvl"]) == False:
            midlevel_terms = dataset.loc[i, "midlvl"]

        lowlevel_terms = re.sub(" ", "_", lowlevel_terms)
        uncontrol_terms = re.sub(" ", "_", uncontrol_terms)
        class_terms = re.sub(" ", "_", class_terms)
        midlevel_terms = re.sub(" ", "_", midlevel_terms)

        # replace spaces with
        lowlevel_terms = re.split(";", lowlevel_terms)
        uncontrol_terms = re.split(";", uncontrol_terms)
        class_terms = re.split(";", class_terms)
        midlevel_terms = re.split(";", midlevel_terms)

        text = (
            str(lowlevel_terms)
            + str(uncontrol_terms)
            + str(class_terms)
            + str(midlevel_terms)
        )
        # text = re.sub('.','_',text)
        text = re.sub("[^_a-zA-Z0-9]", " ", text)
        text = text.lower()
        # text=re.sub("(\\d|\\W)+"," ",text)
        text = text.lstrip()
        text = text.rstrip()
        text = re.sub(" +", " ", text)
        combined_tags.append(text)

        text = str(lowlevel_terms)
        # text = re.sub('.','_',text)
        text = re.sub("[^_a-zA-Z0-9]", " ", text)
        text = text.lower()
        # text=re.sub("(\\d|\\W)+"," ",text)
        text = text.lstrip()
        text = text.rstrip()
        text = re.sub(" +", " ", text)
        idea_tags.append(text)

        text1 = str(uncontrol_terms)
        # text = re.sub('.','_',text)
        text1 = re.sub("[^_a-zA-Z0-9]", " ", text1)
        text1 = text1.lower()
        # text=re.sub("(\\d|\\W)+"," ",text)
        text1 = text1.lstrip()
        text1 = text1.rstrip()
        text1 = re.sub(" +", " ", text1)
        abs_tags.append(text1)
    dataset["combinedTagsNew"] = combined_tags
    dataset["ideaTags"] = idea_tags
    dataset["ideaAbs"] = abs_tags
    skip_index = list(set(skip_index))
    return dataset


def add_combined_tags_new(
    dataset,
    UNCONTROLLED_CODE,
    CLASS_CODE,
    lowlevel_terms,
    uncontrol_terms,
    class_terms,
    midlevel_terms,
):
    print("Adding combined tags new")
    combined_tags = []
    skip_index = []
    # df.drop(columns=['combinedTags'])
    for i in range(0, len(dataset)):
        if pandas.isna(dataset.loc[i, "lowlvl"]) == True:
            lowlevel_terms = ""
            skip_index.append(i)
        if pandas.isna(dataset.loc[i, "lowlvl"]) == False:
            lowlevel_terms = dataset.loc[i, "lowlvl"]

        if pandas.isna(dataset.loc[i, UNCONTROLLED_CODE]) == True:
            uncontrol_terms = ""
            skip_index.append(i)
        if pandas.isna(dataset.loc[i, UNCONTROLLED_CODE]) == False:
            uncontrol_terms = dataset.loc[i, UNCONTROLLED_CODE]

        if pandas.isna(dataset.loc[i, CLASS_CODE]) == True:
            class_terms = ""
            skip_index.append(i)
        if pandas.isna(dataset.loc[i, CLASS_CODE]) == False:
            class_terms = dataset.loc[i, CLASS_CODE]

        if pandas.isna(dataset.loc[i, "midlvl"]) == True:
            midlevel_terms = ""
            skip_index.append(i)
        if pandas.isna(dataset.loc[i, "midlvl"]) == False:
            midlevel_terms = dataset.loc[i, "midlvl"]

        lowlevel_terms = re.sub(" ", "_", lowlevel_terms)
        uncontrol_terms = re.sub(" ", "_", uncontrol_terms)
        class_terms = re.sub(" ", "_", class_terms)
        midlevel_terms = re.sub(" ", "_", midlevel_terms)

        # replace spaces with
        lowlevel_terms = re.split(";", lowlevel_terms)
        uncontrol_terms = re.split(";", uncontrol_terms)
        class_terms = re.split(";", class_terms)
        midlevel_terms = re.split(";", midlevel_terms)

        text = (
            str(lowlevel_terms)
            + str(uncontrol_terms)
            + str(class_terms)
            + str(midlevel_terms)
        )
        # text = re.sub('.','_',text)
        text = re.sub("[^_a-zA-Z0-9]", " ", text)
        text = text.lower()
        # text=re.sub("(\\d|\\W)+"," ",text)
        text = text.lstrip()
        text = text.rstrip()
        text = re.sub(" +", " ", text)
        combined_tags.append(text)

    dataset["combinedTagsNew"] = combined_tags
    skip_index = list(set(skip_index))
    return dataset


def add_ideastag_ideas_abs(dataset, UNCONTROLLED_CODE, lowlevel_terms, uncontrol_terms):
    print("Adding ideas tags new")
    combined_tags = []
    combined_tags1 = []
    skip_index = []
    # df.drop(columns=['combinedTags'])
    for i in range(0, len(dataset)):
        if pandas.isna(dataset.loc[i, "lowlvl"]) == True:
            lowlevel_terms = ""
            skip_index.append(i)
        if pandas.isna(dataset.loc[i, "lowlvl"]) == False:
            lowlevel_terms = dataset["lowlvl"][i]

        if pandas.isna(dataset.loc[i, UNCONTROLLED_CODE]) == True:
            uncontrol_terms = ""
            skip_index.append(i)
        if pandas.isna(dataset.loc[i, UNCONTROLLED_CODE]) == False:
            uncontrol_terms = dataset[UNCONTROLLED_CODE][i]

        lowlevel_terms = re.sub(" ", "_", lowlevel_terms)
        # uncontrol_terms = re.sub(' ', '_',uncontrol_terms)
        # class_terms = re.sub(' ', '_',class_terms)
        # midlevel_terms = re.sub(' ', '_',midlevel_terms)

        # replace spaces with
        lowlevel_terms = re.split(";", lowlevel_terms)
        uncontrol_terms = re.split(";", uncontrol_terms)
        # class_terms = re.split(';', class_terms)
        # midlevel_terms = re.split(';', midlevel_terms)

        text = str(lowlevel_terms)
        # text = re.sub('.','_',text)
        text = re.sub("[^_a-zA-Z0-9]", " ", text)
        text = text.lower()
        # text=re.sub("(\\d|\\W)+"," ",text)
        text = text.lstrip()
        text = text.rstrip()
        text = re.sub(" +", " ", text)
        combined_tags.append(text)

        text1 = str(uncontrol_terms)
        # text = re.sub('.','_',text)
        text1 = re.sub("[^_a-zA-Z0-9]", " ", text1)
        text1 = text1.lower()
        # text=re.sub("(\\d|\\W)+"," ",text)
        text1 = text1.lstrip()
        text1 = text1.rstrip()
        text1 = re.sub(" +", " ", text1)
        combined_tags1.append(text1)

    dataset["ideaTags"] = combined_tags
    dataset["ideaAbs"] = combined_tags1
    skip_index = list(set(skip_index))
    return dataset


def unique_terms(term, test=None):
    print("Calculating unique terms")
    unique_terms = []
    for i in range(0, len(term)):
        set_string = []
        terms = term[i].split(";")
        for j in range(0, len(terms)):
            if test is not None:
                for tst in test:
                    if terms[j] == tst:
                        set_string.append(terms[j])
            else:
                set_string.append(terms[j])
        unique_terms.append(set_string)
    return unique_terms


def sorted_unique_from_list(list):
    return tuple(sorted(set(itertools.chain(*list))))


def create_new_alias(alias, lows, highs, mids, output):
    new_alias = pandas.DataFrame(lows["names"].to_list(), columns=["LowLevel"])
    # new_alias["LowLevel"] = lows["names"]
    new_alias["MidLevel"] = "other"  # lows["names"]
    new_alias["HighLevel"] = "other"  # lows["names"]
    new_alias["freq_low"] = lows["freq"]
    new_alias["freq_mid"] = lows["freq"]
    new_alias["freq_high"] = lows["freq"]
    alias = alias.reset_index()
    new_alias = new_alias.reset_index()
    term1 = new_alias["LowLevel"].to_numpy()
    term2 = alias["LowLevel"].to_numpy()
    matcher = term1[:, None] == term2[None, :]
    i, j = np.where(matcher)
    new_alias.loc[i, "MidLevel"] = alias.loc[j, "MidLevel"].to_list()
    new_alias.loc[i, "HighLevel"] = alias.loc[j, "HighLevel"].to_list()
    new_alias = new_alias_freq(new_alias, lows, "freq_low", "LowLevel")
    new_alias = new_alias_freq(new_alias, mids, "freq_mid", "MidLevel")
    new_alias = new_alias_freq(new_alias, highs, "freq_high", "HighLevel")

    return new_alias


def create_industry(new_alias, dataset):
    industry_terms = (
        new_alias.loc[new_alias["HighLevel"] == "industries", "MidLevel"]
        .unique()
        .tolist()
    )
    industry_df = pandas.DataFrame()
    industry_df["names"] = industry_terms
    ind_freq = []
    for i in range(0, len(industry_df)):
        freq = 0
        term1 = industry_df["names"][i]
        for j in range(0, len(dataset)):
            mid_terms = dataset["MidLevel"][j]
            for k in range(0, len(mid_terms)):
                term2 = mid_terms[k]
                if term1 == term2:
                    freq = freq + 1
        ind_freq.append(freq)
    industry_df["freq"] = ind_freq
    return industry_df


def term_combinations(terms):
    print("calculating term combinations")
    document = terms

    # Get all of the unique entries you have
    varnames = sorted_unique_from_list(document)

    # Get a list of all of the combinations you have
    expanded = [tuple(itertools.combinations(d, 2)) for d in document]
    # print(expanded[:10])
    expanded = itertools.chain(*expanded)

    # Sort the combinations so that A,B and B,A are treated the same
    expanded = [tuple(sorted(d)) for d in expanded]

    # count the combinations
    c = Counter(expanded)

    # Create the table
    table = np.zeros((len(varnames), len(varnames)), dtype=int)

    for i, v1 in enumerate(varnames):
        for j, v2 in enumerate(varnames[i:]):
            j = j + i
            table[i, j] = c[v1, v2]
            table[j, i] = c[v1, v2]
    return table, varnames


def new_alias_freq(new_alias, terms, freq_col, col):
    # if col not in new_alias:
    for i in terms.index:
        term = terms.loc[i, "names"]
        freq = terms.loc[i, "freq"]
        new_alias.loc[new_alias[col] == term, freq_col] = freq
    # new_alias[freq_col] = 0
    # term1 = new_alias[col].to_numpy()
    # term2 = terms["names"].to_numpy()
    # matches = term1[:, None] == term2[None, :]
    # i, j = np.where(matches)
    # new_alias.loc[i, freq_col] = terms.loc[j, "freq"]
    return new_alias


def split_dataset_into_high_level(
    dataset,
    high_level=[
        "business",
        "displays",
        "end users and user experience",
        "industries",
        "technology",
        "use cases",
        "standards",
    ],
):
    # dataset["business"] = np.nan
    # dataset["displays"] = np.nan
    # dataset["end users and user experience"] = np.nan
    # dataset["industries"] = np.nan
    # dataset["technology"] = np.nan
    # dataset["use cases"] = np.nan
    # dataset["standards"] = np.nan
    for hl in high_level:
        dataset[hl] = np.nan
        dataset.loc[dataset["HighLevel"].str.contains(hl, regex=False), hl] = 1

    # for i in range(0, len(dataset)):
    #     for j in range(0, len(dataset["HighLevel"].str.split(";"))):
    #         if dataset["HighLevel"][i][j] == "business":
    #             dataset["business"][i] = 1
    #         if dataset["HighLevel"][i][j] == "displays":
    #             dataset["displays"][i] = 1
    #         if dataset["HighLevel"][i][j] == "end users and user experience":
    #             dataset["end users and user experience"][i] = 1
    #         if dataset["HighLevel"][i][j] == "industries":
    #             dataset["industries"][i] = 1
    #         if dataset["HighLevel"][i][j] == "technology":
    #             dataset["technology"][i] = 1
    #         if dataset["HighLevel"][i][j] == "use cases":
    #             dataset["use cases"][i] = 1
    #         if dataset["HighLevel"][i][j] == "standards":
    #             dataset["standards"][i] = 1
    df_tech = dataset.loc[(dataset["technology"] == 1), ["IdeasTagsPlusAbstract"]]
    df_business = dataset.loc[(dataset["business"] == 1), ["IdeasTagsPlusAbstract"]]
    df_displays = dataset.loc[(dataset["displays"] == 1), ["IdeasTagsPlusAbstract"]]
    df_users = dataset.loc[
        (dataset["end users and user experience"] == 1), ["IdeasTagsPlusAbstract"]
    ]
    df_industries = dataset.loc[(dataset["industries"] == 1), ["IdeasTagsPlusAbstract"]]
    df_usecases = dataset.loc[(dataset["use cases"] == 1), ["IdeasTagsPlusAbstract"]]
    df_standards = dataset.loc[(dataset["standards"] == 1), ["IdeasTagsPlusAbstract"]]
    df0 = pandas.concat([df_tech, df_business, df_displays]).index
    df1 = pandas.concat([df_tech, df_displays]).drop_duplicates().index
    df2 = pandas.concat([df_tech, df_users, df_displays]).drop_duplicates().index
    df3 = pandas.concat([df_tech, df_industries, df_displays]).drop_duplicates().index
    df4 = pandas.concat([df_industries, df_business, df_users]).drop_duplicates().index
    df5 = pandas.concat([df_industries, df_tech, df_usecases]).drop_duplicates().index
    df6 = pandas.concat([df_standards, df_tech]).drop_duplicates().index
    df7 = pandas.concat([df_industries, df_tech, df_business]).drop_duplicates().index
    df8 = pandas.concat([df_tech, df_industries, df_business]).drop_duplicates().index
    slice_sizes = [
        len(df0),
        len(df1),
        len(df2),
        len(df3),
        len(df4),
        len(df5),
        len(df6),
        len(df7),
        len(df8),
    ]
    return [df0, df1, df2, df3, df4, df5, df6, df7, df8], slice_sizes


def term_frequency(table, varnames, json_filename, csv_filename=None):
    json_data = []
    frequency = []

    for i in range(len(table)):
        tmp = {"category": varnames[i]}
        for j in range(len(table)):
            if int(table[i, j]) > 0:
                tmp[varnames[j]] = int(table[i, j])
                frequency.append([varnames[i], varnames[j], table[i, j]])
        json_data.append(tmp)

    with open(json_filename, "w") as fp:
        json.dump(json_data, fp)
    if csv_filename:
        freq_df = pandas.DataFrame(frequency, columns=["Source", "Target", "Freq"])
        freq_df.to_csv(csv_filename, index=True)


def calculate_term_frequency(terms, dataset, col, YEAR):
    print("Creating term frequency table")
    # print(np.array([terms, np.zeros(len(terms))]).T)

    term_df = pandas.DataFrame([[t, 0] for t in terms], columns=["names", "freq"])
    # term_df["names"] = terms
    # term_df["freq"] = np.zeros(len(terms))
    counter = {"freq_count": []}
    years = np.sort(dataset[YEAR].unique())
    dataset[f"tmp{col}"] = "*" + dataset[col] + "*"
    dataset[f"tmp{col}"] = dataset[f"tmp{col}"].str.replace(";", "*;*")
    for year in years:
        term_df[f"freq_{year}"] = np.zeros(len(terms))
        counter[f"freq_count_{year}"] = []
    for i in range(0, len(terms)):
        term_test = terms[i]
        test = dataset[f"tmp{col}"].str.contains("*" + term_test + "*", regex=False)
        for year in years:
            term_df.loc[term_df.index[i], f"freq_{year}"] += (
                test.loc[dataset[YEAR] == year].astype(int).sum()
            )
        term_df.loc[term_df.index[i], "freq"] += test.astype(int).sum()
    dataset = dataset.drop(f"tmp{col}", axis=1)
    return term_df


def string_column_to_numpy(column, df):
    data = ""  # baseline["Acad Sim Scores"]
    for i in range(0, len(df)):
        data += df.loc[i, column] + ","
    data = data.replace("[", "")
    data = data.replace("]", "")
    data = data.replace(",,", ",")
    data = data.split(",")
    data = [d for d in data if d != ""]
    # data.remove("")
    data = np.array(data)
    data = data.astype(float)
    return data


def calculate_similarities(dataset, baseline, ideas, output):
    docs = ideas["TagsPlusAbstract"].tolist()
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf = vectorizer.fit_transform(docs)
    sim_scores = (tfidf * tfidf.T).A
    ideas["Sim Scores"] = ""
    for i in range(ideas.shape[0]):
        ideas.loc[i, "Sim Scores"] = np.array2string(sim_scores[i, :], separator=",")
    # ideas["Sim Scores"] = sim_scores

    ideas["Mean Sim"] = np.mean(sim_scores, axis=1)
    ideas["Stdev Sim"] = np.std(sim_scores, axis=1)
    ideas["Max Score"] = np.max(sim_scores, axis=1)

    dfs, slice_sizes = split_dataset_into_high_level(dataset)
    means_slice = []
    stdevs_slice = []
    maxes_slice = []

    docs = baseline["Academic"].to_list()
    docs.extend(dataset["TagsPlusAbstract"].to_list())
    vectorizer = TfidfVectorizer(max_features=3000)

    tfidf = vectorizer.fit_transform(docs)
    sim_scores = (tfidf * tfidf.T).A[
        : len(baseline["Academic"].to_list()), len(baseline["Academic"].to_list()) :
    ]
    baseline["Acad Sim Scores"] = sim_scores
    baseline["Acad Mean Sim"] = np.mean(sim_scores, axis=1)
    baseline["Acad Stdev Sim"] = np.std(sim_scores, axis=1)
    baseline["Acad Max Score"] = np.max(sim_scores, axis=1)

    docs = baseline["Fiction"].to_list()
    docs.extend(dataset["TagsPlusAbstract"].to_list())
    vectorizer = TfidfVectorizer(max_features=3000)

    tfidf = vectorizer.fit_transform(docs)
    sim_scores = (tfidf * tfidf.T).A[
        : len(baseline["Fiction"].to_list()), len(baseline["Fiction"].to_list()) :
    ]
    baseline["Fict Sim Scores"] = sim_scores
    baseline["Fict Mean Sim"] = np.mean(sim_scores, axis=1)
    baseline["Fict Stdev Sim"] = np.std(sim_scores, axis=1)
    baseline["Fict Max Score"] = np.max(sim_scores, axis=1)

    dfs, slice_sizes = split_dataset_into_high_level(dataset)
    means_slice = []
    stdevs_slice = []
    maxes_slice = []

    docs = baseline["Academic"].to_list()
    docs.extend(dataset["TagsPlusAbstract"].to_list())
    vectorizer = TfidfVectorizer(max_features=3000)

    tfidf = vectorizer.fit_transform(docs)
    sim_scores = (tfidf * tfidf.T).A[
        : len(baseline["Academic"].to_list()), len(baseline["Academic"].to_list()) :
    ]
    baseline["Acad Sim Scores"] = sim_scores
    baseline["Acad Mean Sim"] = np.mean(sim_scores, axis=1)
    baseline["Acad Stdev Sim"] = np.std(sim_scores, axis=1)
    baseline["Acad Max Score"] = np.max(sim_scores, axis=1)

    docs = baseline["Fiction"].to_list()
    docs.extend(dataset["TagsPlusAbstract"].to_list())
    vectorizer = TfidfVectorizer(max_features=3000)

    tfidf = vectorizer.fit_transform(docs)
    sim_scores = (tfidf * tfidf.T).A[
        : len(baseline["Fiction"].to_list()), len(baseline["Fiction"].to_list()) :
    ]
    baseline["Fict Sim Scores"] = sim_scores
    baseline["Fict Mean Sim"] = np.mean(sim_scores, axis=1)
    baseline["Fict Stdev Sim"] = np.std(sim_scores, axis=1)
    baseline["Fict Max Score"] = np.max(sim_scores, axis=1)

    ind_dfs = dfs  # [df0, df1, df2, df3, df4, df5, df6, df7, df8]
    docs = ideas["TagsPlusAbstract"].to_list()
    docs.extend(dataset["IdeasTagsPlusAbstract"].to_list())

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs)
    idea_indexes = list(np.arange(len(ideas), dtype=int))
    for i, indexes in enumerate(ind_dfs):
        all_indexes = idea_indexes + list(indexes)
        sim_scores = (tfidf[all_indexes, :] * tfidf[all_indexes, :].T).A
        sim_scores = sim_scores[len(ideas) :, : len(ideas)]
        ideas[f"sim_score_mean_{i}"] = np.mean(sim_scores, axis=0)
        ideas[f"sim_score_max_{i}"] = np.max(sim_scores, axis=0)
        ideas[f"sim_score_std_{i}"] = np.std(sim_scores, axis=0)


def find_relevant_citation_to_ideas(ideas, dataset):
    relevant_citations = []
    for j in range(0, len(ideas)):
        x = ideas["Sim Scores"][j]
        temp = pandas.DataFrame()
        temp["scores"] = x
        temp = temp.sort_values("scores", ascending=False)
        index = temp[:5].index.values
        papers = ""
        for i in range(0, len(index)):
            paper = dataset["Citation"][index[i]]
            if papers == "":
                papers = paper
            if papers != "":
                papers = papers + " | " + paper
        relevant_citations.append(papers)
        # print(dataset['Citation'][index[i]])
    ideas["Relevant Publications"] = relevant_citations


def find_top20(dataset, column, newcolumn):
    docs = dataset[column].to_list()
    vectorizer = TfidfVectorizer(max_features=3000)

    tfidf = vectorizer.fit_transform(docs)
    sim_scores = (tfidf * tfidf.T).A

    top_20 = []
    for i in range(len(sim_scores)):
        a = sim_scores[i]
        l = OrderedDict(sorted(enumerate(a), key=itemgetter(1), reverse=True))
        d = dict(sorted(l.items(), key=lambda x: x[1], reverse=True)[1:21])
        keys_list = list(d)
        top_20.append(keys_list)
    dataset[newcolumn] = top_20

    return dataset


def filter_idea_similarity_scores(ideas):
    filtered_scores = []
    for i in range(0, len(ideas)):
        filtered_scores_ = []
        score = ideas["Sim Scores"][i]
        for j in range(0, len(score)):
            try:
                if score[j] in [".", ",", ""]:
                    continue
                if float(score[j]) > 0.058817154045084:
                    filtered_scores_.append(score[j])
            except Exception as e:
                pass
        filtered_scores.append(filtered_scores_)
    ideas["Acad Filtered Scores"] = filtered_scores

    filtered_scores = []
    for i in range(0, len(ideas)):
        filtered_scores_ = []
        score = ideas["Sim Scores"][i]
        for j in range(0, len(score)):
            try:
                if score[j] in [".", ",", "", " "]:
                    continue
                if float(score[j]) > 0.03083764926999431:
                    filtered_scores_.append(score[j])
            except Exception as e:
                pass
        filtered_scores.append(filtered_scores_)
    ideas["Fict Filtered Scores"] = filtered_scores
    return ideas


def calculate_idea_score(ideas):
    points = []

    for i in range(0, len(ideas)):
        scores = ideas["Acad Filtered Scores"][i]
        point = 0
        for j in range(0, len(scores)):
            try:
                if scores[j] in [".", ",", ""]:
                    continue
                score = float(scores[j])
                if score < 0.075:
                    point += 1
                if score < 0.1 and score >= 0.075:
                    point = point + 2
                if score < 0.125 and score >= 0.1:
                    point = point + 3
                if score < 0.15 and score >= 0.125:
                    point = point + 4
                if score < 0.175 and score >= 0.15:
                    point = point + 5
                if score < 0.2 and score >= 0.175:
                    point = point + 6
                if score < 0.225 and score >= 0.2:
                    point = point + 7
                if score < 0.25 and score >= 0.225:
                    point = point + 8
                if score < 0.275 and score >= 0.25:
                    point = point + 9
                if score >= 0.275:
                    point = point + 10
            except Exception as e:
                pass

        points.append(point)
    ideas["Acad Points"] = points

    ideas["Score"] = round(5 * ideas["Acad Points"] / max(ideas["Acad Points"]), 2)
    return ideas


def visualisation_df(varnames, table, name):
    pairs = []
    for i in range(0, len(varnames)):
        tst = varnames[i]
        for j in range(0, len(varnames)):
            tst2 = varnames[j]
            pair = [tst, tst2]
            pairs.append(pair)

    co_value = []
    for i in range(0, len(table)):
        for j in range(0, len(table)):
            co_value.append(table[i, j])

    df_viz = pandas.DataFrame(pairs, columns=[f"{name}1", f"{name}2"])
    df_viz["value"] = co_value
    return df_viz


def calculate_mid_ind_fequency(mids, inds):
    freqs = {}
    for col in mids.columns:
        if "freq" in col:
            freqs[col] = []
    for i in range(0, len(inds)):
        term1 = inds["names"][i]
        for j in range(0, len(mids)):
            term2 = mids["names"][j]
            if term1 == term2:
                for k in freqs.keys():
                    print(mids[k][j], k, j, term1, term2)
                    freqs[k].append(mids[k][j])

    for k in freqs:
        inds.loc[:, k] = freqs[k]
