# Count how often Magnus Carlsen played each opening.
#
# Reads games from two sources:
#   - a PGN file of his professional games
#   - the Chess.com games JSON written by retrieve.py
# takes the ECO code from each game's headers, maps it to an opening name, and
# writes a CSV of openings sorted by how many games he played them in, with a
# count per source.
#
# ECO names come from the Lichess opening tables in data/eco/ (CC0, from
# https://github.com/lichess-org/chess-openings). Those tables list many named
# lines per ECO code; each code is named after its shortest line, which is the
# position that defines the code.
#
# Usage:
#   python3 openings.py                                  # default paths
#   python3 openings.py games.pgn games.json out.csv

import collections
import csv
import datetime
import glob
import json
import os
import re
import sys

PGN_PATH = os.path.join("data", "Carlsen professional games.pgn")
JSON_PATH = os.path.join("data", "carlsen online chess.com games.json")
OUTPUT_PATH = os.path.join("data", "carlsen_openings.csv")
ECO_DIR = os.path.join("data", "eco")

# PGN: matches "Carlsen,M" and "Carlsen,Magnus" but not other Carlsens (e.g. "Carlsen,H").
PGN_PLAYER_PATTERN = re.compile(r"^Carlsen,\s*M")
CHESSCOM_USERNAME = "magnuscarlsen"

# Lichess names A00 after one arbitrary irregular first move; the code really
# covers all of them.
NAME_OVERRIDES = {"A00": "Uncommon Opening"}

TAG_PATTERN = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$', re.MULTILINE)


def load_eco_names(eco_dir):
    """Return {ECO code: opening name}, using each code's shortest line."""
    shortest = {}
    for path in sorted(glob.glob(os.path.join(eco_dir, "*.tsv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                eco = row["eco"]
                if eco not in shortest or len(row["pgn"]) < len(shortest[eco]["pgn"]):
                    shortest[eco] = row

    names = {eco: row["name"] for eco, row in shortest.items()}
    names.update(NAME_OVERRIDES)
    return names


def read_pgn_games(pgn_path):
    """Yield (ECO code, color, year) for each game in the PGN file.

    Color is "white", "black", or None if Carlsen isn't found in the game.
    Year is an int, or None if the Date tag has no year.
    """
    headers = {}

    def finish(headers):
        if PGN_PLAYER_PATTERN.match(headers.get("White", "")):
            color = "white"
        elif PGN_PLAYER_PATTERN.match(headers.get("Black", "")):
            color = "black"
        else:
            color = None
        year = headers.get("Date", "")[:4]
        return headers.get("ECO", ""), color, int(year) if year.isdigit() else None

    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            match = TAG_PATTERN.match(line)
            if not match:
                continue
            tag, value = match.groups()
            # Every game starts with an Event tag.
            if tag == "Event" and headers:
                yield finish(headers)
                headers = {}
            headers[tag] = value
    if headers:
        yield finish(headers)


def read_chesscom_games(json_path):
    """Yield (ECO code, color, year) for each standard-chess game in the JSON file.

    Chess960 and other variants are skipped: Chess.com still tags them with
    ECO codes, but those don't describe an opening when the pieces start
    shuffled.
    """
    with open(json_path, encoding="utf-8") as f:
        games = json.load(f)

    for game in games:
        if game.get("rules") != "chess":
            continue
        headers = dict(TAG_PATTERN.findall(game.get("pgn", "")))
        if game["white"]["username"].lower() == CHESSCOM_USERNAME:
            color = "white"
        elif game["black"]["username"].lower() == CHESSCOM_USERNAME:
            color = "black"
        else:
            color = None
        year = datetime.datetime.fromtimestamp(game["end_time"], datetime.timezone.utc).year
        yield headers.get("ECO", ""), color, year


def main():
    pgn_path = sys.argv[1] if len(sys.argv) > 1 else PGN_PATH
    json_path = sys.argv[2] if len(sys.argv) > 2 else JSON_PATH
    output_path = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_PATH

    eco_names = load_eco_names(ECO_DIR)

    totals = collections.Counter()
    by_color = {"white": collections.Counter(), "black": collections.Counter()}
    by_source = {"pgn": collections.Counter(), "chesscom": collections.Counter()}
    skipped = collections.Counter()

    sources = [
        ("pgn", read_pgn_games(pgn_path)),
        ("chesscom", read_chesscom_games(json_path)),
    ]
    for source, games in sources:
        for eco, color, _year in games:
            eco = eco.strip().upper()
            if eco not in eco_names:
                skipped[source] += 1
                continue
            totals[eco] += 1
            by_source[source][eco] += 1
            if color:
                by_color[color][eco] += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ECO", "Opening", "Games", "As White", "As Black", "PGN Games", "Chess.com Games"])
        # Most-played first, ties broken by ECO code.
        for eco, count in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([
                eco,
                eco_names[eco],
                count,
                by_color["white"][eco],
                by_color["black"][eco],
                by_source["pgn"][eco],
                by_source["chesscom"][eco],
            ])

    print(f"Counted {sum(totals.values())} games across {len(totals)} openings")
    print(f"  PGN:       {sum(by_source['pgn'].values())} games")
    print(f"  Chess.com: {sum(by_source['chesscom'].values())} games")
    for source, count in skipped.items():
        print(f"Skipped {count} {source} games with a missing or unrecognized ECO code")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
