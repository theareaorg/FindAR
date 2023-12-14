import json
import numpy as np


def create_sankey(alias, output):
    alias = alias.copy()
    # add __ to lowlevel and _ to mid to make sure rows aren't duplicated
    alias.loc[:, "LowLevel"] += "__"
    alias.loc[:, "MidLevel"] += "_"
    alias["cat"] = "Category"
    mids = [
        alias.loc[
            alias["MidLevel"] == term,
            ["MidLevel", "HighLevel", "freq_mid", "freq_mid"],
        ].values.tolist()[0]
        for term in alias["MidLevel"].unique()
    ]

    highs = [
        alias.loc[
            alias["HighLevel"] == term,
            ["HighLevel", "cat", "freq_high", "freq_high"],
        ].values.tolist()[0]
        for term in alias["HighLevel"].unique()
    ]

    lows = alias[["LowLevel", "MidLevel", "freq_low", "freq_low"]].values.tolist()
    sankey = ["Category", None, int(alias["freq_low"].sum()), -1] + highs + lows + mids
    # sankey = []
    # # "Name", "Parent", "Value", "Color"
    # sankey.append(["Category", None, int(alias["freq_low"].sum()), -1])
    # hl_map = {}
    # for h in highs:
    #     sankey.append

    # for i, hl in enumerate(alias["HighLevel"].unique()):
    #     # add all of the high level terms with the
    #     count = alias.loc[alias["HighLevel"] == hl, "freq_high"].to_list()[0]
    #     print(hl, count)
    #     if not count or np.isnan(count):
    #         count = 0
    #     else:
    #         try:
    #             count = int(count)
    #         except:
    #             print(count)
    #             print(np.isnan(count))
    #             count = 0
    #     sankey.append(
    #         [
    #             hl,
    #             "Category",
    #             count,
    #             i,
    #         ]
    #     )
    #     hl_map[hl] = i
    # for ml in alias["MidLevel"].unique():
    #     # add all of the mid level terms with the
    #     count = alias.loc[alias["MidLevel"] == ml, "freq_mid"].to_list()[0]
    #     if not count or np.isnan(count):
    #         count = 0
    #     else:
    #         try:
    #             count = int(count)
    #         except:
    #             print(count)
    #             print(np.isnan(count))
    #             count = 0
    #     sankey.append(
    #         [
    #             ml,
    #             alias.loc[alias["MidLevel"] == ml, "HighLevel"].to_list()[0],
    #             count,
    #             hl_map[alias.loc[alias["MidLevel"] == ml, "HighLevel"].to_list()[0]],
    #         ]
    #     )
    # for ll in alias["LowLevel"].unique():
    #     # add all of the low level terms with the
    #     count = alias.loc[alias["LowLevel"] == ll, "freq_low"].to_list()[0]
    #     if not count or np.isnan(count):
    #         count = 0
    #     else:
    #         try:
    #             count = int(count)
    #         except Exception as e:
    #             print(e)
    #             print(count)
    #             print(np.isnan(count))
    #             count = 0
    #     sankey.append(
    #         [
    #             ll,
    #             alias.loc[alias["LowLevel"] == ll, "MidLevel"].to_list()[0],
    #             count,
    #             hl_map[alias.loc[alias["LowLevel"] == ll, "HighLevel"].to_list()[0]],
    #         ]
    #     )

    with open(f"{output}/json/sankey-data.json", "w") as f:
        json.dump(sankey, f)


# run test on alias.csv if this is run as a script
if __name__ == "__main__":
    import pandas as pd

    alias = pd.read_csv("alias.csv")
    create_sankey(alias, ".")
