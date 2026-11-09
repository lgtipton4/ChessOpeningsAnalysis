# Chart how Magnus Carlsen's openings changed over the years, using the
# move-based classification instead of ECO headers.
#
# Reads data/carlsen_games_clustered.csv (written by match_openings.py and then
# cluster_openings.py) and draws three charts into charts/:
#
#   theory_by_year.png      Left: mean book depth (plies spent inside named
#                           lines) per year. Right: share of games that follow
#                           a named line's exact move order. One line per
#                           source, since OTB and online games differ a lot.
#
#   clusters_by_year.png    Heatmap of the share of each year's games that fall
#                           in each k-means opening cluster, with Carlsen as
#                           White and as Black side by side.
#
#   diversity_by_year.png   How varied each year's openings were: the
#                           effective number of clusters, exp(entropy) of that
#                           year's cluster shares. It equals k when every
#                           cluster is played equally and 1 when only one is.
#
# Years with fewer than MIN_GAMES games in a series are left out, since a
# handful of games makes a share or mean meaningless.
#
# Usage:
#   python3 opening_trends.py                       # clusters from PGN games
#   python3 opening_trends.py --source Chess.com    # clusters from online games
#   python3 opening_trends.py --source all

import argparse
import collections
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

INPUT_PATH = os.path.join("data", "carlsen_games_clustered.csv")
CHARTS_DIR = "charts"
THEORY_OUTPUT_PATH = os.path.join(CHARTS_DIR, "theory_by_year.png")
CLUSTERS_OUTPUT_PATH = os.path.join(CHARTS_DIR, "clusters_by_year.png")
DIVERSITY_OUTPUT_PATH = os.path.join(CHARTS_DIR, "diversity_by_year.png")

MIN_GAMES = 30

SOURCE_COLORS = {"PGN": "#2a78d6", "Chess.com": "#eb6834"}
SOURCE_LABELS = {"PGN": "Professional (PGN)", "Chess.com": "Online (Chess.com)"}
# Sequential blue ramp; zero recedes into the chart surface.
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "share", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
TEXT_PRIMARY = "#1f1f1d"
TEXT_SECONDARY = "#5f5e58"
GRID = "#e6e5e0"


def load_games(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [g for g in csv.DictReader(f) if g["Year"] and g["MatchType"] != "none"]


def style_axis(ax):
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def plot_theory(games, output_path):
    by_series = collections.defaultdict(list)
    for g in games:
        by_series[(g["Source"], int(g["Year"]))].append(g)

    fig, (depth_ax, exact_ax) = plt.subplots(1, 2, figsize=(14, 5.5))
    for source, color in SOURCE_COLORS.items():
        years = sorted(y for s, y in by_series if s == source and len(by_series[(s, y)]) >= MIN_GAMES)
        depth = [np.mean([int(g["BookDepth"]) for g in by_series[(source, y)]]) for y in years]
        exact = [100 * np.mean([g["MatchType"] == "exact" for g in by_series[(source, y)]]) for y in years]
        for ax, values in ((depth_ax, depth), (exact_ax, exact)):
            ax.plot(years, values, color=color, linewidth=2, marker="o", markersize=5,
                    label=SOURCE_LABELS[source])

    depth_ax.set_title("Plies played inside named opening lines (mean)",
                       loc="left", fontsize=12, color=TEXT_PRIMARY)
    exact_ax.set_title("Games following a named line's exact move order (%)",
                       loc="left", fontsize=12, color=TEXT_PRIMARY)
    exact_ax.set_ylim(0, 100)
    for ax in (depth_ax, exact_ax):
        style_axis(ax)
    depth_ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("How deep into known theory Magnus Carlsen's games go", x=0.01, ha="left",
                 fontsize=14, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


def plot_clusters(games, source, output_path):
    if source != "all":
        games = [g for g in games if g["Source"] == source]
    cluster_names = {}
    for g in games:
        cluster_names[int(g["Cluster"])] = g["ClusterName"]
    clusters = sorted(cluster_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), sharey=True)
    image = None
    for ax, color in zip(axes, ("white", "black")):
        counts = collections.Counter((int(g["Year"]), int(g["Cluster"])) for g in games if g["Color"] == color)
        totals = collections.Counter(int(g["Year"]) for g in games if g["Color"] == color)
        years = sorted(y for y, n in totals.items() if n >= MIN_GAMES)
        shares = np.array([[counts[(y, c)] / totals[y] for y in years] for c in clusters])
        image = ax.imshow(shares, cmap=HEATMAP_CMAP, vmin=0, vmax=0.3, aspect="auto")
        ax.set_xticks(range(len(years)), years, rotation=45, color=TEXT_SECONDARY, fontsize=8)
        ax.set_title(f"Carlsen as {color.capitalize()}", loc="left", fontsize=12, color=TEXT_PRIMARY)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(clusters)), [cluster_names[c] for c in clusters],
                       color=TEXT_SECONDARY, fontsize=8)

    bar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.01)
    bar.set_label("Share of that year's games (darkest = 30% or more)", color=TEXT_SECONDARY)
    bar.outline.set_visible(False)
    source_label = "all games" if source == "all" else f"{source} games"
    fig.suptitle(f"Magnus Carlsen's opening clusters by year ({source_label})", x=0.01, ha="left",
                 fontsize=14, color=TEXT_PRIMARY)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart to {output_path}")


def effective_clusters(games):
    """Return exp(Shannon entropy) of the games' cluster shares."""
    counts = np.array(list(collections.Counter(g["Cluster"] for g in games).values()))
    shares = counts / counts.sum()
    return float(np.exp(-(shares * np.log(shares)).sum()))


def plot_diversity(games, output_path):
    by_series = collections.defaultdict(list)
    for g in games:
        by_series[(g["Source"], int(g["Year"]))].append(g)

    fig, ax = plt.subplots(figsize=(10, 5))
    for source, color in SOURCE_COLORS.items():
        years = sorted(y for s, y in by_series if s == source and len(by_series[(s, y)]) >= MIN_GAMES)
        ax.plot(years, [effective_clusters(by_series[(source, y)]) for y in years], color=color,
                linewidth=2, marker="o", markersize=5, label=SOURCE_LABELS[source])

    n_clusters = len({g["Cluster"] for g in games})
    ax.set_ylim(0, n_clusters)
    ax.set_title(f"Effective number of opening clusters played per year (out of {n_clusters})",
                 loc="left", fontsize=12, color=TEXT_PRIMARY)
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["all", "PGN", "Chess.com"], default="PGN",
                        help="games to use for the cluster heatmap")
    parser.add_argument("--input", default=INPUT_PATH)
    args = parser.parse_args()

    games = load_games(args.input)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plot_theory(games, THEORY_OUTPUT_PATH)
    plot_clusters(games, args.source, CLUSTERS_OUTPUT_PATH)
    plot_diversity(games, DIVERSITY_OUTPUT_PATH)


if __name__ == "__main__":
    main()
