import pandas
import os
from area.sankey import create_sankey
import seaborn as sns
import nltk
from datetime import datetime

nltk.download("stopwords")


nltk.download("wordnet")

# from sklearn.cross_validation import train_test_split

import os
from scipy.stats import norm

# import active_learning
from area.process import *
from area.inputoutput import *


def process_dataset(
    CLASS_CODE="Classification_index_terms",
    UNCONTROLLED_CODE="Uncontrolled_terms",
    CONTROLLED_CODE="Controlled_terms",
    YEAR="Year",
    VENUE="Journal_name",
    data_directory="storage/data",
    output_folder_name="storage/output/output",
    datafile_name="current_data.csv",
    baseline_academic_filename="sampleACAD.txt",
    baseline_fiction_filename="sampleFICT.txt",
    alias_filename="term-bucketing.csv",
    replacement_filename="replacements-new.csv",
    edges_filename="data_lab_test.csv",
    baseline_filename="baseline.csv",
    ideas_filename="research_topics.csv",
    plot=True,
):
    now = datetime.now()
    date = now.strftime("%B_%Y")

    # os.chdir(data_directory)
    # output = "output_2"
    # create a new directory for each run
    # output = increment_path(output_folder_name, mkdir=True)
    if not os.path.exists(output_folder_name):
        os.mkdir(output_folder_name)
    if not os.path.exists(f"{output_folder_name}/output_{date}"):
        os.mkdir(f"{output_folder_name}/output_{date}")
    output = f"{output_folder_name}/output_{date}"
    os.mkdir(f"{output}/figures")
    os.mkdir(f"{output}/json")
    os.mkdir(f"{output}/text")

    print(f"Saving outputs to {output}")

    # load the dataset

    dataset = pandas.read_csv(f"{data_directory}/{datafile_name}")
    dataset = dataset.reset_index()  # loc[:100, :]
    alias = pandas.read_csv(
        f"{data_directory}/{alias_filename}"
    )  # "term-bucketing.csv")
    replace = pandas.read_csv(
        f"{data_directory}/{replacement_filename}"
    )  # "replacements-new.csv")
    edges = pandas.read_csv(
        f"{data_directory}/{edges_filename}"
    )  # "data_lab_test.csv")
    baseline = pandas.read_csv(
        f"{data_directory}/{baseline_filename}"
    )  # "basline.csv")
    ideas = pandas.read_csv(f"{data_directory}/{ideas_filename}", encoding="ISO-8859-1")
    with open(f"{data_directory}/{baseline_academic_filename}") as f:
        baselineAcad = f.readlines()
    f.close()
    with open(f"{data_directory}/{baseline_fiction_filename}") as f:
        baselineFict = f.readlines()
    f.close()

    # remove duplicates from alias
    alias = preprocess_alias(alias)
    # remove punctuations, numbers and special characters
    baselineAcad = process_text(baselineAcad)
    baselineFict = process_text(baselineFict)

    edges = map_edges_to_alias(edges, alias)
    control_keywords, uncontrol_keywords, classification_code = codes_from_dataset(
        dataset,
        CLASS_CODE=CLASS_CODE,
        CONTROLLED_CODE=CONTROLLED_CODE,
        UNCONTROLLED_CODE=UNCONTROLLED_CODE,
    )

    dataset = add_lowlevel(
        dataset, CONTROLLED_CODE=CONTROLLED_CODE, replacement_terms=replace
    )
    dataset = apply_term_bucketing(
        dataset,
        alias,
    )
    dataset = combined_tags(
        dataset,
        UNCONTROLLED_CODE=UNCONTROLLED_CODE,
        CLASS_CODE=CLASS_CODE,
    )
    dataset["cleanAbstract"] = process_text(dataset["Abstract"].astype(str).to_list())
    dataset["TagsPlusAbstract"] = (
        dataset["combinedTagsNew"] + " " + dataset["cleanAbstract"]
    )
    dataset["IdeasTagsPlusAbstract"] = (
        dataset["cleanAbstract"] + dataset["ideaTags"] + dataset["ideaAbs"]
    )
    dataset.dropna(subset=["cleanAbstract"], inplace=True)
    dataset["Relevant"] = dataset["Relevant"].astype(int)
    # dataset.to_csv("storage/for_learning.csv", index=False)
    # dataset, change_log = active_learning.active_learning(
    #     dataset, "IdeasTagsPlusAbstract", mode="min"
    # )

    #################################
    ## IDEAS ########################
    ideas["abstract"] = (
        ideas["Title"]
        + " "
        + ideas["Description"]
        + " "
        + ideas["Importance"]
        + " "
        + ideas["Keywords"]
        + " "
        + ideas["Stakeholders"]
        + " "
        + ideas["Methdologies"]
        + " "
        + ideas["Program"]
    )
    ideas["tags"] = ideas["Keywords"]
    ideas["cleanAbstract"] = process_text(ideas["abstract"].to_list())
    ideas["cleanTags"] = (
        ideas["tags"]
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(",", ";")
        .replace(";", " ")
    )
    ideas["tagsWC"] = len(ideas["cleanTags"].str.split(" "))
    ideas["abstractWC"] = len(ideas["cleanAbstract"].str.split(" "))
    for i in ideas.index:
        ideas.loc[i, "abstractWC"] = len(ideas.loc[i, "cleanAbstract"].split(" "))
        ideas.loc[i, "tagsWC"] = len(ideas.loc[i, "cleanTags"].split(";"))

    ideas["combinedAbsandTags"] = ideas["cleanTags"] + ideas["cleanAbstract"]
    dataset["highlvl"] = dataset["HighLevel"]
    for i in dataset.index:
        dataset.loc[i, "highlvl"] = ";".join(
            list(set(dataset.loc[i, "highlvl"].split(";")))
        )

    midlvl = unique_terms(dataset["midlvl"])
    lowlvl = unique_terms(dataset["lowlvl"])
    highlvl = unique_terms(dataset["highlvl"])

    sorted_unique_from_list(highlvl)

    table_high, varnames_high = term_combinations(highlvl)
    table_mid, varnames_mid = term_combinations(midlvl)
    table_low, varnames_low = term_combinations(lowlvl)
    ## calculate the frequency of each term for low/mid/high
    mids = calculate_term_frequency(
        sorted_unique_from_list(midlvl), dataset, "midlvl", YEAR
    )
    mids.to_csv(f"{output}/text/mid_level_frequency.csv")
    lows = calculate_term_frequency(
        sorted_unique_from_list(lowlvl), dataset, "lowlvl", YEAR
    )
    lows.to_csv(f"{output}/text/low_level_frequency.csv")

    highs = calculate_term_frequency(
        sorted_unique_from_list(highlvl), dataset, "highlvl", YEAR
    )
    highs.to_csv(f"{output}/text/high_level_frequency.csv")

    new_alias = create_new_alias(alias, lows, highs, mids, output=output)

    industry_df = create_industry(new_alias, dataset=dataset)

    ilvl = unique_terms(dataset["MidLevel"], industry_df["names"])
    table_i, varnames_i = term_combinations(ilvl)

    # add frequencies of terms into dataframe

    create_sankey(new_alias, output)

    df_viz_high = visualisation_df(varnames_high, table_high, "High")
    df_viz_ind = visualisation_df(varnames_i, table_i, "Ind")
    df_viz = visualisation_df(varnames_mid, table_mid, "Mid")
    df_viz_low = visualisation_df(varnames_low, table_low, "Low")

    dataset.loc[dataset["Relevant"] == 1, "Title"] = (
        "*" + dataset.loc[dataset["Relevant"] == 1, "Title"]
    )

    inds = pandas.DataFrame(list(set(df_viz_ind["Ind1"])), columns=["names"])

    calculate_mid_ind_fequency(mids, inds)

    ideas["hlvl"] = ideas["Categories"]

    new_terms = []
    ideas["hlvl"] = ideas["hlvl"].str.lower().str.replace(", ", ";")
    for i in range(0, len(ideas)):
        terms_h = ideas["hlvl"][i].split(";")
        for j in range(0, len(terms_h)):
            if terms_h[j] == "display devices":
                terms_h[j] = "displays"
            if terms_h[j] == "end users":
                terms_h[j] = "end users and user experience"
        new_terms.append(terms_h)
    ideas["newTags"] = new_terms
    # run_plots()
    from area.plotting import (
        plot_industry_term_per_year,
        plot_term_pairs_scatterplot,
        plot_standards,
        plot_frequency_bar_chart,
        plot_similarity_wordcount,
    )

    if plot:
        plot_industry_term_per_year(inds, output)

        plot_term_pairs_scatterplot(
            df_viz,
            output,
            x="Mid1",
            y="Mid2",
            figname=f"midlvl_term_pairs_scatterplot.png",
            slicev=50,
        )
        plot_term_pairs_scatterplot(
            df_viz_ind,
            output,
            x="Ind1",
            y="Ind2",
            figname=f"indlvl_term_pairs_scatterplot.png",
        )
        plot_term_pairs_scatterplot(
            df_viz_low,
            output,
            x="Low1",
            y="Low2",
            figname=f"lowlvl_term_pairs_scatterplot.png",
            slicev=50,
        )
        plot_term_pairs_scatterplot(
            df_viz_high,
            output,
            x="High1",
            y="High2",
            figname=f"highvl_term_pairs_scatterplot",
            slicev=40,
        )

    if plot:
        df_table_mid = pandas.DataFrame(table_mid)

        try:
            sns.clustermap(
                df_table_mid, xticklabels=varnames_mid, yticklabels=varnames_mid
            )
        except Exception as e:
            print("Clustermap for mid level failed")
            print(e)

        # sns.heatmap(table, xlabel=midlvl_names)

        df_table_low = pandas.DataFrame(table_low)
        try:
            sns.clustermap(
                df_table_low, xticklabels=varnames_low, yticklabels=varnames_low
            )
        except Exception as e:
            print("Clustermap for low level failed")
            print(e)

    # Example data
    mids = mids.sort_values(by=["freq"], ascending=False)
    highs = highs.sort_values(by=["freq"], ascending=False)
    lows = lows.sort_values(by=["freq"], ascending=False)
    industry_df = industry_df.sort_values(by=["freq"], ascending=False)

    df_instance = highs.sort_values(by=["freq"], ascending=False)

    plot_standards(df_instance, output)

    plot_frequency_bar_chart(lows, output=output)

    ideas["TagsPlusAbstract"] = ideas["cleanAbstract"] + ideas["cleanTags"]
    calculate_similarities(dataset, baseline, ideas, output)

    plot_similarity_wordcount(ideas, wc="tagsWC", output=output)
    plot_similarity_wordcount(ideas, wc="abstractWC", output=output)

    ideas.to_csv(f"{output}/text/ideas-results.csv")
    ideas.rename(
        columns={"Mean Sim": "Mean", "Stdev Sim": "Stdev", "Max Score": "Max"}
    ).to_csv(f"{output}/text/ideas_scored.csv")

    baseline = pandas.read_csv(f"{data_directory}/{baseline_filename}")

    data = string_column_to_numpy("Acad Sim Scores", baseline)
    # raise Exception

    mu, sd = norm.fit(data)
    data = string_column_to_numpy("Fict Sim Scores", baseline)

    mu, sd = norm.fit(data)

    import ast

    new_string = []
    for i in range(0, len(ideas)):
        string = ideas["Sim Scores"][i]
        string = ast.literal_eval(string)
        new_string.append(string)
    ideas["Sim Scores (new)"] = new_string

    ideas["Sim Scores"] = ideas["Sim Scores"].str.replace("[", "").str.replace("]", "")
    ideas = filter_idea_similarity_scores(ideas)
    ideas = calculate_idea_score(ideas)

    ideas.to_csv("newScoresForIdeas.csv")

    find_relevant_citation_to_ideas(ideas, dataset)
    ideas.to_csv("newScoresForIdeas.csv")
    export_scored_topics(ideas, date)

    ideas["Sim Scores"] = new_string

    ideas.drop(
        columns=[
            "Sim Scores",
            "Sim Scores (new)",
            "Acad Filtered Scores",
            "Fict Filtered Scores",
        ]
    )
    ideas.to_csv("ideas_final2.csv")

    find_top20(dataset, column="cleanAbstract", newcolumn="Top20Abs")
    find_top20(dataset, column="combinedTagsNew", newcolumn="Top20Tags")
    find_top20(dataset, column="TagsPlusAbstract", newcolumn="Top20AbsAndTags")
    dataset.to_csv(f"{output}/dataset_{date}.csv")
    export_analysis(
        dataset,
        output,
        YEAR,
        VENUE,
        varnames_low,
        varnames_mid,
        table_low,
        table_mid,
        alias,
        new_alias,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_directory", type=str, default="data", help="Specify the data directory"
    )
    parser.add_argument(
        "--output_folder_name",
        type=str,
        default="output",
        help="Specify the output folder name",
    )
    parser.add_argument(
        "--datafile_name",
        type=str,
        default="current_data.csv",
        help="Specify the data file name",
    )
    parser.add_argument(
        "--baseline_academic_filename",
        type=str,
        default="sampleACAD.txt",
        help="Specify the baseline academic filename",
    )
    parser.add_argument(
        "--baseline_fiction_filename",
        type=str,
        default="sampleFICT.txt",
        help="Specify the baseline fiction filename",
    )
    parser.add_argument(
        "--alias_filename",
        type=str,
        default="term-bucketing.csv",
        help="Specify the alias filename",
    )
    parser.add_argument(
        "--replacement_filename",
        type=str,
        default="replacements-new.csv",
        help="Specify the replacement filename",
    )
    parser.add_argument(
        "--lows_filename",
        type=str,
        default="lows.csv",
        help="Specify the lows filename",
    )
    parser.add_argument(
        "--mids_filename",
        type=str,
        default="mids.csv",
        help="Specify the mids filename",
    )
    parser.add_argument(
        "--highs_filename",
        type=str,
        default="highs.csv",
        help="Specify the highs filename",
    )
    parser.add_argument(
        "--edges_filename",
        type=str,
        default="data_lab_test.csv",
        help="Specify the edges filename",
    )
    parser.add_argument(
        "--baseline_filename",
        type=str,
        default="baseline.csv",
        help="Specify the baseline filename",
    )
    parser.add_argument(
        "--ideas_filename",
        type=str,
        default="research_topics.csv",
        help="Specify the ideas filename",
    )
    parser.add_argument(
        "--plot", type=bool, default=True, help="Specify whether to plot or not"
    )
    args = parser.parse_args()

    process_dataset(
        CLASS_CODE="Classification_index_terms",
        UNCONTROLLED_CODE="Uncontrolled_terms",
        CONTROLLED_CODE="Controlled_terms",
        YEAR="Year",
        VENUE="Journal_name",
        data_directory=args.data_directory,
        output_folder_name=args.output_folder_name,
        datafile_name=args.datafile_name,
        baseline_academic_filename=args.baseline_academic_filename,
        baseline_fiction_filename=args.baseline_fiction_filename,
        alias_filename=args.alias_filename,
        replacement_filename=args.replacement_filename,
        edges_filename=args.edges_filename,
        baseline_filename=args.baseline_filename,
        ideas_filename=args.ideas_filename,
        plot=args.plot,
    )
