import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


def calc_mean_similarity(dataset):
    vectorizer = TfidfVectorizer()
    docs = dataset.tolist()
    tfidf = vectorizer.fit_transform(docs)
    sim_scores = (tfidf * tfidf.T).A
    return np.mean(sim_scores, axis=1)


def calc_min_similarity(dataset_label, dataset_unlabelled):
    vectorizer = TfidfVectorizer()
    docs = dataset_label.tolist()
    docs.extend(dataset_unlabelled.tolist())
    tfidf = vectorizer.fit_transform(docs)
    print("tfod", tfidf.shape)
    sim_scores = (tfidf * tfidf.T).A
    # print(sim_scores.min(), sim_scores.max())
    sim_scores = sim_scores[0 : dataset_label.shape[0], dataset_label.shape[0] :]
    print(sim_scores.shape)
    print(np.min(sim_scores, axis=0).shape)
    return np.nanmax(sim_scores, axis=0)


def threshold_function(iteration_number, initial_threshold, decay_rate):
    return initial_threshold * np.exp(decay_rate * iteration_number)


def active_learning(
    dataset, col, update_number=500, mode="min", max_iter=1e3, threshold=0.02
):
    print('Beginning active learning with mode "{}"'.format(mode))
    run = True
    i = 0
    last_change = 1
    change_log = []
    true_labels = dataset.index[dataset["Relevant"] == 1]
    false_labels = dataset.index[dataset["Relevant"] == 0]
    true_scores = calc_min_similarity(
        dataset.loc[true_labels, col], dataset.loc[true_labels, col]
    )
    false_scores = calc_min_similarity(
        dataset.loc[false_labels, col], dataset.loc[false_labels, col]
    )
    true_scores_false = calc_min_similarity(
        dataset.loc[true_labels, col], dataset.loc[false_labels, col]
    )
    false_scores_true = calc_min_similarity(
        dataset.loc[false_labels, col], dataset.loc[true_labels, col]
    )
    plt.hist(true_scores, bins=25, density=True, alpha=0.6, color="g")
    plt.hist(true_scores_false, bins=25, density=True, alpha=0.6, color="r")

    plt.savefig("true_scores.png")
    plt.clf()
    plt.hist(false_scores, bins=25, density=True, alpha=0.6, color="g")
    plt.hist(false_scores_true, bins=25, density=True, alpha=0.6, color="r")

    plt.savefig("false_scores.png")
    plt.clf()

    return
    while run:
        threshold = threshold_function(i, threshold, 0.01)
        print("Iteration {} threshold {}".format(i, threshold))

        # print("Threshold: {}".format(threshold))
        true_labels = dataset.index[dataset["Relevant"] == 1]
        false_labels = dataset.index[dataset["Relevant"] == 0]
        if mode == "min":
            true_scores = calc_min_similarity(
                dataset.loc[true_labels, col], dataset[col]
            )
            false_scores = calc_min_similarity(
                dataset.loc[false_labels, col], dataset[col]
            )
            new_true = dataset.index[(true_scores > threshold)].to_numpy()
            new_false = dataset.index[(false_scores > threshold)].to_numpy()
            # print((dataset.loc[new_true, "Relevant"] == -1).index)
            # print((dataset.loc[new_false, "Relevant"] == -1).index)
            # print(dataset.loc[dataset["Relevant"] == -1,])
            dataset.loc[(dataset.loc[new_true, "Relevant"] == -1).index, "Relevant"] = 1

            dataset.loc[
                (dataset.loc[new_false, "Relevant"] == -1).index, "Relevant"
            ] = 0
            # dataset.loc[dataset.loc[new_true, "Relevant"] = 1
            # unlabelled_index_false = dataset.loc[
            #     np.logical_and(dataset['Relevant']==-1, (true_scores > threshold)), dataset["Relevant"] == -1
            # ].index
            # unlabelled_index_true = dataset.loc[
            #     new_false, dataset["Relevant"] == 1
            # ].index
            # new_true = new_true.to_numpy()
            # new_false = new_false.to_numpy()
            change = (
                np.abs(
                    (true_scores > threshold).astype(int)
                    - (dataset["Relevant"] == 1).astype(int)
                ).sum()
                + np.abs(
                    (false_scores > threshold).astype(int)
                    - (dataset["Relevant"] == 0).astype(int)
                ).sum()
            ) / dataset.shape[0]
            np.random.shuffle(new_true)
            np.random.shuffle(new_false)

            # dataset.loc[unlabelled_index_false, "Relevant"] = 0
            # dataset.loc[unlabelled_index_true, "Relevant"] = 1
            dataset.loc[new_true[:update_number], "Relevant"] = 1
            dataset.loc[new_false[:update_number], "Relevant"] = 0
            i += 1
            change_log.append([i, change])
            unlabelled = (dataset["Relevant"] == -1).sum()
            if unlabelled > 0:
                print(f"Still have {unlabelled} unlabelled samples")
                continue
            if np.isclose(change, last_change):
                run = False
                print("Converged after {} iterations".format(i))
            if i > max_iter:
                run = False
                print("Max iterations reached")
            last_change = change

            print("Change in labels: {}".format(change))
            # dataset.loc[true_labels,'min_sim'] = true_scores
        # elif mode == 'mean':
    return dataset, np.array(change_log)


# main function to run code
if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("for_learning.csv", encoding="ISO-8859-1")
    print(data.columns)
    data.dropna(subset=["IdeasTagsPlusAbstract"], inplace=True)
    data["Relevant"] = data["Relevant"].astype(int)
    print(data.columns)
    dataset, change_log = active_learning(data, "IdeasTagsPlusAbstract", mode="min")
    dataset.to_csv("dataset_learnt.csv", index=False)
    plt.figure()
    plt.plot(change_log[:, 0], change_log[:, 1])
