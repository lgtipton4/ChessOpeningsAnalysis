# Count how often Magnus Carlsen played each opening, per year and per source.
#
# Uses the same game sources and ECO names as openings.py, and writes one row
# per (year, source, ECO code) so the most-used opening of each year can be
# charted.
#
# Usage:
#   python3 openings_by_year.py                 # default output path
#   python3 openings_by_year.py out.csv

import collections
import csv
import os
import sys

from openings import (
    ECO_DIR,
    JSON_PATH,
    PGN_PATH,
    load_eco_names,
    read_chesscom_games,
    read_pgn_games,
)

OUTPUT_PATH = os.path.join("data", "carlsen_openings_by_year.csv")


def opening_family(name):
    """'Sicilian Defense: Najdorf Variation' -> 'Sicilian Defense'."""
    return name.split(":")[0]


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else OUTPUT_PATH

    eco_names = load_eco_names(ECO_DIR)
    counts = collections.Counter()

    sources = [
        ("PGN", read_pgn_games(PGN_PATH)),
        ("Chess.com", read_chesscom_games(JSON_PATH)),
    ]
    for source, games in sources:
        for eco, _color, year in games:
            eco = eco.strip().upper()
            if eco in eco_names and year:
                counts[(year, source, eco)] += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Year", "Source", "ECO", "Opening", "Family", "Games"])
        for (year, source, eco), count in sorted(counts.items()):
            name = eco_names[eco]
            writer.writerow([year, source, eco, name, opening_family(name), count])

    print(f"Wrote {len(counts)} rows to {output_path}")


if __name__ == "__main__":
    main()
