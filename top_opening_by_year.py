# Chart Magnus Carlsen's most-played opening for each year.
#
# Reads the per-year counts written by openings_by_year.py and draws one bar
# per year. The bar's height is the top opening's share of that year's games,
# and its color says which opening it was.
#
# Usage:
#   python3 top_opening_by_year.py                     # all sources, by family
#   python3 top_opening_by_year.py --source PGN        # only OTB games
#   python3 top_opening_by_year.py --level opening     # specific ECO names
#   python3 top_opening_by_year.py --output chart.png

import argparse
import collections
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INPUT_PATH = os.path.join("data", "carlsen_openings_by_year.csv")
OUTPUT_PATH = os.path.join("data", "top_opening_by_year.png")

# Fixed categorical order; openings get colors in order of first appearance.
PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#7f63d6", "#9a6b3c",
]
OTHER_COLOR = "#a3a29b"


def top_openings(path, source, level):
    """Return [(year, opening, games, year_total)] sorted by year."""
    column = "Family" if level == "family" else "Opening"
    counts = collections.Counter()
    totals = collections.Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if source != "all" and row["Source"] != source:
                continue
            year, games = int(row["Year"]), int(row["Games"])
            counts[(year, row[column])] += games
            totals[year] += games

    best = {}
    for (year, opening), games in counts.items():
        if year not in best or games > best[year][1]:
            best[year] = (opening, games)
    return [(year, *best[year], totals[year]) for year in sorted(best)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["all", "PGN", "Chess.com"], default="all")
    parser.add_argument("--level", choices=["family", "opening"], default="family")
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    rows = top_openings(args.input, args.source, args.level)
    if not rows:
        raise SystemExit("No games found for that selection.")

    colors = {}
    for _, opening, _, _ in rows:
        if opening not in colors:
            slot = len(colors)
            colors[opening] = PALETTE[slot] if slot < len(PALETTE) else OTHER_COLOR

    years = [r[0] for r in rows]
    shares = [100 * games / total for _, _, games, total in rows]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(years, shares, width=0.8, color=[colors[r[1]] for r in rows],
                  edgecolor="white", linewidth=2)

    # Label each bar with the raw game count so the share has context.
    for bar, (_, _, games, _) in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(games), ha="center", va="bottom",
                fontsize=8, color="#5f5e58")

    source_label = "all games" if args.source == "all" else f"{args.source} games"
    ax.set_title(f"Magnus Carlsen's most-played opening each year ({source_label})",
                 loc="left", fontsize=14, color="#1f1f1d")
    ax.set_ylabel("Share of that year's games (%)", color="#5f5e58")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45, colors="#5f5e58")
    ax.tick_params(axis="y", colors="#5f5e58")
    ax.set_ylim(0, max(shares) * 1.15)
    ax.grid(axis="y", color="#e6e5e0")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [Patch(color=c, label=name) for name, c in colors.items()]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              bbox_to_anchor=(1.0, 1.0), fontsize=9)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved chart to {args.output}")
    for year, opening, games, total in rows:
        print(f"{year}  {opening}  ({games} of {total} games)")


if __name__ == "__main__":
    main()
