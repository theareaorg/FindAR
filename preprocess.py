import pandas as pd
import os
import json
import argparse
from datetime import date


def preprocess(new_data_path: str, original_data_path: str, output_path: str):
    """Merge newly downloaded data with the original dataset

    Parameters
    ----------
    new_data_path : str
        new downloaded data
    original_data_path : str
        existing dataset to merge with
    output_path : str
        where to store the merged dataset
    """
    data = pd.read_csv(new_data_path)
    ## extract out year as the 4 consecutive digits
    data = data.drop(data.index[data["DOI"].isna()])
    data["Year"] = data["Year"].str.extract(r"(\d\d\d\d)")
    data["PN"] = ""
    for i in data.index:
        try:
            data.loc[i, "PN"] = json.loads(data.loc[i, "prop"]).get("PN", "")
        except Exception as e:
            print(e)
            continue
        # print(json.loads(data.loc[i, "prop"])["PN"])
        # raise Exception
    # data.to_csv(f"{output_path}/dataset_merged2.csv", index=False)

    original = pd.read_csv(original_data_path)
    publisher_list = list(original["PN"].unique())
    data = data.loc[data["PN"].isin(publisher_list), :]
    data = data.loc[~data["DOI"].isin(original["DOI"]), :]
    data = data.drop(columns=["prop", "obj"])
    data["Relevant"] = -1
    # data.drop()
    # data = pd.concat([da/ta], ignore_index=True)
    data.to_csv(
        f"{output_path}/dataset_{date.today().year}_{date.today().month}_{date.today().day}.csv",
        index=False,
    )
    # print(json.loads(data.l/oc[i, "prop"].replace("'", '"'))["PN"])
    data.to_csv(f"{output_path}/current_data.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print("running preprocess.py")
    parser.add_argument(
        "--new_data_path", type=str, default="storage/data/new_dataset.csv"
    )
    parser.add_argument(
        "--original_data_path", type=str, default="storage/data/existing_dataset.csv"
    )
    parser.add_argument("--output_path", type=str, default="storage")
    args = parser.parse_args()
    data_path = args.new_data_path
    original_data_path = args.original_data_path
    output_path = args.output_path
    preprocess(data_path, original_data_path, output_path)
    print("preprocess.py finished")
