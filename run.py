from area import SearchEV
import pandas as pd
import time

from area.download import download_documents

searcher = SearchEV(
    # startyear=2022,
    updateNumber=4
)
all_results, flag = searcher.search("augmented reality")
print(f"Downloading all records")
download_start_time = time.time()
df = pd.DataFrame(all_results, columns=["id"])

df = download_documents(list(df["id"].unique()))
# raise BaseException
print(f"Downloading {len(all_results)} took {download_start_time-time.time()} seconds")
# print("Finished downloading records")
df.to_csv("storage/dataset.csv")
