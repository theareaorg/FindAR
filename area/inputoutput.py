from pathlib import Path
import os
import json
import pandas
import re
from .process import term_frequency


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def export_analysis(
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
):
    df_json = (
        dataset[
            [
                "Title",
                "DOI",
                YEAR,
                "Abstract",
                "lowlvl",
                "midlvl",
                "highlvl",
                VENUE,
                "Top20AbsAndTags",
                "Top20Abs",
                "Top20Tags",
                "Citation",
            ]
        ]
        .copy()
        .rename(
            columns={
                "lowlvl": "LowLevel",
                "highlvl": "HighLevel",
                "midlvl": "MidLevel",
                VENUE: "Venue",
                YEAR: "Publication year",
            }
        )
    )  # pandas.DataFrame()
    df_json = df_json.sort_values(by="Publication year", ascending=False)

    with open(f"{output}/text/low_terms.txt", "w") as f:
        for item in varnames_low:
            f.write("%s\n" % item)

    df = dataset.copy()
    df["LowLevel"] = df["lowlvl"]
    df["MidLevel"] = df["midlvl"]

    a = df_json.to_json(orient="records")
    with open(f"{output}/json/data.json", "w") as f:
        f.write(a)
    df_json.to_csv(f"{output}/text/df_json.csv")

    lowlvl = []
    for i in range(0, len(dataset)):
        terms = str(dataset["lowlvl"][i])
        terms = re.split(";", terms)
        lowlvl.append(terms)

    AllLowLvlTerms = []
    for i in range(0, len(lowlvl)):
        for j in range(0, len(lowlvl[i])):
            text = lowlvl[i][j]
            AllLowLvlTerms.append(text)

    b = list(set(AllLowLvlTerms))

    df = pandas.DataFrame()
    df["LowLevel"] = b
    counter = len(df)
    freq_count = []
    mid_levels = []

    for i in range(0, len(df)):
        freq_count.append(
            dataset["lowlvl"].str.contains(df.loc[i, "LowLevel"], regex=False).sum()
        )

    # df['MidLevel']=mid_levels
    df["freq"] = freq_count

    for i in range(0, len(df["LowLevel"])):
        string1 = df["LowLevel"][i]
        for j in range(0, len(alias["LowLevel"])):
            string2 = alias["LowLevel"][j]
            if string1 == string2:
                mid_levels.append(alias["MidLevel"][j])
    term_frequency(
        table_low,
        varnames_low,
        f"{output}/json/terms-ll.json",
        f"{output}/text/low.csv",
    )
    term_frequency(
        table_mid,
        varnames_mid,
        f"{output}/json/terms.json",
        f"{output}/text/mid.csv",
    )
    new_alias = new_alias.sort_values(by="LowLevel", ascending=True)
    buckets = []
    for i in range(0, len(varnames_mid)):
        mid_string = varnames_mid[i]
        terms = []
        for j in range(0, len(new_alias)):
            if new_alias["MidLevel"][j] == str(mid_string):
                terms.append(new_alias["LowLevel"][j])
        buckets.append(terms)

    topics_json = []
    new_alias_ = new_alias.sort_values(by="LowLevel", ascending=True)
    new_alias_.to_csv(f"{output}/text/new_alias.csv")
    file1 = open(f"{output}/json/category.json", "w")
    file1.write("[")
    for i in range(0, len(new_alias_)):
        file1.write("{" + '"label"' + ":" + '"' + new_alias_["LowLevel"][i] + '",')
        file1.write('"category"' + ":" + '"' + new_alias_["MidLevel"][i] + '"')
        file1.write("},")
    # file1.write('{"category": ""}')
    file1.write("]")
    for i in range(0, len(varnames_mid)):
        topics_json.append({"term": varnames_mid[i], "lowLevel": buckets[i]})

    with open(f"{output}/json/topics.json", "w") as fp:
        json.dump(topics_json, fp)


def export_scored_topics(ideas, date):
    print(ideas.columns)
    ideas[
        [
            "Title",
            "Description",
            "Keywords",
            "Categories",
            "Stakeholders",
            "Importance",
            "Methdologies",
            "Timeframes",
            "Program",
            "Miscellaneous",
            "Author",
            "History",
            "Score",
            "Relevant Publications",
        ]
    ].to_csv(f"scored_topics_{date}.csv")
