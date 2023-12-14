from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

colors = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffffff",
    "#000000",
]


def plot_frequency_bar_chart(lows, output):
    print("Plotting frequency bar chart")
    lows = lows.sort_values(by=["freq"], ascending=False).reset_index()
    max_freq = lows["freq"].max()
    n_pages = np.ceil(len(lows) / 62).astype(int)
    fig, ax = plt.subplots()

    for j in range(0, n_pages):
        fig.clear()
        print(f"Page {j + 1} of {n_pages}")
        page = j + 1

        plt.rcdefaults()

        # ax.plot([1,3,2])

        set_size(5, 10)
        plt.gcf().subplots_adjust(left=0.50)

        first_term = 62 * j
        last_term = first_term + 61
        if last_term > len(lows):
            last_term = len(lows) - 1
        low_terms = lows.loc[first_term:last_term, "names"].to_list()
        low_freq = lows.loc[first_term:last_term, "freq"].to_list()
        y_pos = np.arange(len(low_terms))

        rects = ax.barh(y_pos, low_freq)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(low_terms)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel("Count")
        ax.set_xlim(0, max_freq)
        ax.set_title("Frequency of Low-Level Terms")
        for i, v in enumerate(low_freq):
            ax.text(v, i + 0.25, str(v), color="black", fontweight="normal")

        # for index, value in enumerate(mid_freq):
        # plt.text(value, index, str(value))

        # plt.figure(figsize=(1,100))
        fig.savefig(f"{output}/figures/low_page" + str(page) + ".pdf")


def plot_term_pairs_scatterplot(
    df_viz,
    output,
    x="Mid1",
    y="Mid2",
    a4_dims=(10, 10),
    figname=None,
    slicev=None,
):
    print(f"Plotting term pair scatter plot {x} {y}")
    df_viz = df_viz.sort_values(by="value", ascending=False)

    if slicev == None:
        slicev = len(df_viz)
    fig, ax = plt.subplots(figsize=a4_dims)
    customPalette = sns.set_palette(sns.color_palette(colors), n_colors=13)
    sp = sns.scatterplot(
        ax=ax,
        data=df_viz[:slicev],
        x=x,
        y=y,
        hue="value",
        size="value",
        sizes=(50, 250),
        legend=False,
        palette=cmap,
    )
    sp.tick_params(axis="both", which="major", labelsize=10)
    plt.xticks(rotation=90)
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    # plt.tight_layout()
    if figname:
        print(f"Saving figure {figname}.png")
        fig.savefig(f"{output}/figures/{figname}", bbox_inches="tight")


def set_size(w, h, ax=None):
    """w, h: width, height in inches"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def plot_industry_term_per_year(inds, output):
    print("Plotting industry term per year")
    inds = inds.sort_values(by="freq", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 10))
    years = []
    for col in inds.columns:
        if "freq_" in col:
            years.append(col.replace("freq_", ""))

    # sum_2020_2021 = sum(inds["freq_2020"])+sum(inds["freq_2021"])

    for i in range(0, len(inds)):
        x_values = years  # [2017, 2018, 2019, 2020]
        y_values = [
            inds.loc[i, f"freq_{year}"] / inds[f"freq_{year}"].sum() for year in years
        ]
        # y_values = [inds["freq_2017"][i]/sum(inds["freq_2017"]), inds["freq_2018"][i]/sum(inds["freq_2018"]),
        #             inds["freq_2019"][i]/sum(inds["freq_2019"]),
        #             (inds["freq_2020"][i]+inds["freq_2021"][i])/sum_2020_2021]
        ax.plot(x_values, y_values, label=inds["names"][i])

    ax.set_xticks(years)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), prop=fontP)
    ax.set_title("Relative contribution of industry term per year")
    fig.savefig(f"{output}/figures/industry_term_per_year.png", bbox_inches="tight")


def plot_standards(df_instance, output):
    y_pos = np.arange(len(df_instance))
    fig, ax = plt.subplots(figsize=(5, 5))
    rects = ax.barh(y_pos, df_instance["freq"], color=qualitative_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_instance["names"])
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.xlim([0, max(df_instance["freq"])])
    ax.set_xlabel("Count")
    ax.set_title("Frequencies")
    for i, v in enumerate(df_instance["freq"]):
        ax.text(v, i + 0.25, str(int(v)), color="black", fontweight="normal")

    fig.savefig(f"{output}/figures/standards.png", dpi=1000)


# qualitative_colors = sns.color_palette(colors)
# sns.palplot(qualitative_colors)

import matplotlib.colors as clr

cmap = clr.LinearSegmentedColormap.from_list(
    "area_colors", ["#31A6B2", "#083A49"], N=256
)
# customPalette = sns.set_palette(sns.color_palette(colors),n_colors=13)

font = {"family": "serif", "weight": "normal", "size": 10}

plt.rc("font", **font)


from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size("small")


qualitative_colors = sns.color_palette(colors)
# customPalette = sns.set_palette(sns.color_palette(colors), n_colors=13)
# plt.clf()


def plot_similarity_wordcount(ideas, wc, output):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ideas["Mean Sim"], ideas[wc])
    if wc == "tagsWC":
        ax.set_xlabel("Mean of cosine similarity scores")
        ax.set_ylabel("Number of tags on research concepts")
        fig.savefig(f"{output}/figures/tagsWCvsMeanSim.png")
    if wc == "abstractWC":
        ax.set_xlabel("Mean of cosine similarity scores")
        ax.set_ylabel("Word count of research concepts")
        fig.savefig(f"{output}/figures/abstractWCvsMeanSim.png")


def plot_idea_similarity_distribution(ideas, output):
    # Fit a normal distribution to the data:
    for i in range(0, len(ideas)):
        data = ideas["Sim Scores"][i]
        mu, std = norm.fit(data)

        # Plot the histogram.
        plt.hist(data, bins=25, density=True, alpha=0.6, color="g")

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        plt.xlim([0, 0.35])
        plt.ylim([0, 35])
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, "k", linewidth=2)
        title = "RC #" + str(i) + " (mu = %.2f,  std = %.2f)" % (mu, std)

        plt.title(title)
        plt.savefig(f"{output}/rc{i}_pdf.png")
        plt.clf()


def plot_pdfs():
    mu = np.mean(data)
    std = np.std(data)
    cdf = norm.cdf(data, loc=mu, scale=std)
    cdf_val = []
    for i in range(0, len(cdf)):
        cdf_val.append(cdf[i])
    cdf_values = pandas.DataFrame()
    cdf_values["academic scores"] = data
    cdf_values["cdf"] = cdf_val

    # norm.ppf(cdf(0.99))
    counter = 0
    for i in range(0, len(cdf)):
        if cdf[i] > 0.99:
            counter = counter + 1
    # print(counter)

    list1 = cdf.tolist()

    # cdf.size

    # defining the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # No of Data points
    N = len(data)

    # initializing random values
    # data = np.random.randn(N)

    # getting data of the histogram
    count, bins_count = np.histogram(data, bins=10)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)

    # plotting PDF and CDF
    plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.legend()

    data = string_column_to_numpy("Acad Sim Scores", baseline)

    # data = baseline["Acad Sim Scores"][0]
    # for i in range(1, len(baseline)):
    #     data = data + baseline["Acad Sim Scores"][i]

    mu, std = norm.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color="g")

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    plt.xlim([0, 0.35])
    plt.ylim([0, 35])
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2)
    title = "Academic" + " (mu = %.2f,  std = %.2f)" % (mu, std)

    plt.title(title)
    plt.savefig(f"{output}/academic_pdf.png")
    plt.clf()

    # plt.show()

    data = string_column_to_numpy("Fict Sim Scores", baseline)

    # data = baseline["Fict Sim Scores"][0]
    # for i in range(1, len(baseline)):
    #     data = data + baseline["Fict Sim Scores"][i]

    mu, std = norm.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color="g")

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    plt.xlim([0, 0.35])
    plt.ylim([0, 35])
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2)
    title = "Fiction" + " (mu = %.2f,  std = %.2f)" % (mu, std)

    plt.title(title)
    plt.savefig(f"{output}/fiction_pdf.png")
    plt.clf()
