from area import SearchEV
import pandas as pd
import time
from datetime import datetime

from area.download import parallel_download_documents


def download(
    updateNumber=4, startyear=None, endyear=None, output_path="storage/dataset.csv"
):
    dfs = []
    if updateNumber in [1, 2, 3, 4]:
        searcher = SearchEV(
            updateNumber=updateNumber,
        )
        all_results, flag = searcher.search("augmented reality")
        print(f"Downloading all records")
        download_start_time = time.time()
        df = pd.DataFrame(all_results, columns=["id"])

        df = parallel_download_documents(list(df["id"].unique()))
        dfs.append(df)
    else:
        if startyear is None:
            raise ValueError("startyear must be specified")
        if endyear is None:
            endyear = datetime.now().year

        for year in range(startyear, endyear + 1, 1):
            doctypemap = {"ca": "conference", "ja": "journal"}
            for dt in ["ca", "ja"]:
                print(f"Downloading all {doctypemap[dt]} for {year}")
                searcher = SearchEV(
                    startyear=year,
                    endyear=year,
                    updateNumber=updateNumber,
                )
                all_results, flag = searcher.search(
                    "augmented reality", document_type=dt
                )
                print(f"{len(all_results)} records found for year {year}")
                df = pd.DataFrame(all_results, columns=["id"])

                dfs.append(df)
                # raise BaseException

    print(f"Downloading all records")
    download_start_time = time.time()
    df = pd.concat(dfs, ignore_index=True)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse

    print("Running downloader")
    parser = argparse.ArgumentParser()
    parser.add_argument("--startyear", type=int, default=None)
    parser.add_argument("--updateNumber", type=int, default=4)
    parser.add_argument("--endyear", type=int, default=None)

    parser.add_argument("--output_path", type=str, default="storage/dataset.csv")
    args = parser.parse_args()
    download(
        startyear=args.startyear,
        updateNumber=args.updateNumber,
        output_path=args.output_path,
        endyear=args.endyear,
    )
    print("Finished downloading")
