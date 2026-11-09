# Classify each of Magnus Carlsen's games by its actual moves, using
# Levenshtein (edit) distance to the named opening lines.
#
# The ECO header on a game says which opening it was filed under, not how
# closely the players followed it. At top level, games rarely play a named line
# to its end: players transpose move orders, insert a waiting move, or leave
# theory early. So instead of trusting the header, each game is compared move
# by move with all ~3,900 named lines in the Lichess tables.
#
# Nearest named line. Each game is matched in two complementary ways, and the
# better-scoring match wins:
#
#   * Edit distance: for a line of m plies, the Levenshtein distance d between
#     the line and the game's first m plies counts the moves that would have to
#     be inserted, deleted, or substituted to turn one into the other. One
#     deviating move costs 1; an inserted move pair costs 2. Each line is scored
#     m - 2d, so a long line that the game mostly followed beats a short line it
#     followed exactly. Lines where more than MAX_EDIT_FRACTION of the moves
#     differ are ruled out.
#
#   * Transposition: edit distance penalizes a different move order heavily
#     even when it reaches the very same position (1.d4 g6 2.e4 is a Modern
#     Defense just like 1.e4 g6 2.d4). So if the game ever reaches the position
#     a line ends in, that line counts as fully followed (d = 0, score m).
#
#   The game takes the name and ECO code of its best line. MatchType says how
#   it was reached ("exact", "transposition", or "approximate"), and
#   adherence = 1 - d/m says how closely the game followed it.
#
# Book depth. How many plies the game stays inside the named lines, i.e. how
# many of its opening positions occur somewhere in the Lichess tables. Comparing
# positions rather than move orders means a transposition doesn't count as
# leaving book. This measures how far into known theory each game went.
#
# Moves are compared as whole tokens, not characters: each distinct SAN move
# is mapped to a single Unicode character so rapidfuzz's fast C++ Levenshtein
# can run on the move sequences.
#
# Usage:
#   python3 match_openings.py                 # default output path
#   python3 match_openings.py out.csv

import collections
import csv
import os
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from games import load_opening_lines, opening_family, position_keys, read_all_records

OUTPUT_PATH = os.path.join("data", "carlsen_games_openings.csv")

# A line is only a candidate if at most this fraction of its moves differ.
MAX_EDIT_FRACTION = 0.25
# Each edit costs this many plies of score (see the header comment).
EDIT_PENALTY = 2

# Start of the CJK block: tens of thousands of consecutive ordinary
# characters, so no SAN move is ever mapped to a surrogate or control code.
FIRST_CODEPOINT = 0x4E00


class MoveEncoder:
    """Map each distinct SAN move to one character, so sequences become strings."""

    def __init__(self):
        self.codes = {}

    def encode(self, moves):
        chars = []
        for move in moves:
            if move not in self.codes:
                self.codes[move] = chr(FIRST_CODEPOINT + len(self.codes))
            chars.append(self.codes[move])
        return "".join(chars)


def rank(m, dist):
    """Sort key for a candidate line: higher score first, then fewer edits.

    Distances never exceed the longest line, so multiplying the score by 100
    keeps the two apart.
    """
    return (m - EDIT_PENALTY * dist) * 100.0 - dist


def nearest_by_edit_distance(game_strings, line_strings):
    """Return (best line index, edit distance, rank) arrays; index -1 if none qualifies.

    Lines are grouped by length m so every game can be truncated to m plies
    and compared with all lines of that length in one vectorized cdist call.
    """
    n_games = len(game_strings)
    best_rank = np.full(n_games, -np.inf)
    best_line = np.full(n_games, -1)
    best_dist = np.full(n_games, -1)

    by_length = collections.defaultdict(list)
    for index, line in enumerate(line_strings):
        by_length[len(line)].append(index)

    cols = np.arange(n_games)
    for m, indices in sorted(by_length.items()):
        truncated = [game[:m] for game in game_strings]
        dist = cdist([line_strings[i] for i in indices], truncated,
                     scorer=Levenshtein.distance, workers=-1, dtype=np.int32)
        key = rank(m, dist)
        key[dist > MAX_EDIT_FRACTION * m] = -np.inf

        row = key.argmax(axis=0)
        better = key[row, cols] > best_rank
        best_rank[better] = key[row, cols][better]
        best_line[better] = np.asarray(indices)[row[better]]
        best_dist[better] = dist[row, cols][better]

    return best_line, best_dist, best_rank


def nearest_by_position(game_positions, lines, line_positions):
    """Return, per game, the index of the longest line whose final position it reaches (or -1)."""
    line_ending_at = {}
    for index, positions in enumerate(line_positions):
        end = positions[-1]
        if end not in line_ending_at or len(lines[index]["moves"]) > len(lines[line_ending_at[end]]["moves"]):
            line_ending_at[end] = index

    best = []
    for positions in game_positions:
        reached = [line_ending_at[p] for p in positions if p in line_ending_at]
        best.append(max(reached, key=lambda i: len(lines[i]["moves"]), default=-1))
    return best


def book_depths(game_positions, line_positions):
    """Return, per game, how many leading plies reach positions found in named lines."""
    book = {p for positions in line_positions for p in positions}
    depths = []
    for positions in game_positions:
        depth = 0
        while depth < len(positions) and positions[depth] in book:
            depth += 1
        depths.append(depth)
    return depths


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else OUTPUT_PATH

    lines = load_opening_lines()
    records = [r for r in read_all_records() if r["moves"]]

    encoder = MoveEncoder()
    line_strings = [encoder.encode(line["moves"]) for line in lines]
    game_strings = [encoder.encode(r["moves"]) for r in records]
    line_positions = [position_keys(line["moves"]) for line in lines]
    game_positions = [position_keys(r["moves"]) for r in records]

    depths = book_depths(game_positions, line_positions)
    edit_line, edit_dist, edit_rank = nearest_by_edit_distance(game_strings, line_strings)
    position_line = nearest_by_position(game_positions, lines, line_positions)

    match_types = collections.Counter()
    agree_code = agree_group = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Source", "GameID", "Year", "Color", "TimeClass", "HeaderECO", "Moves",
            "BookDepth", "MatchType", "MatchedECO", "MatchedOpening", "MatchedFamily",
            "LineLength", "EditDistance", "Adherence",
        ])
        for i, record in enumerate(records):
            line_index, dist = edit_line[i], edit_dist[i]
            # A transposition counts as following the line exactly (d = 0).
            if position_line[i] >= 0 and rank(len(lines[position_line[i]]["moves"]), 0) > edit_rank[i]:
                line_index, dist = position_line[i], 0

            if line_index >= 0:
                line = lines[line_index]
                m = len(line["moves"])
                if record["moves"][:m] == line["moves"]:
                    match_type = "exact"
                elif dist == 0:
                    match_type = "transposition"
                else:
                    match_type = "approximate"
                match = [match_type, line["eco"], line["name"], opening_family(line["name"]),
                         m, int(dist), round(1 - dist / m, 3)]
                match_types[match_type] += 1
                agree_code += line["eco"] == record["eco"]
                agree_group += line["eco"][:2] == record["eco"][:2]
            else:
                match = ["none", "", "", "", "", "", ""]
                match_types["none"] += 1

            writer.writerow([
                record["source"], record["game_id"], record["year"] or "",
                record["color"] or "", record["time_class"], record["eco"],
                " ".join(record["moves"]), depths[i], *match,
            ])

    matched = len(records) - match_types["none"]
    print(f"Matched {matched} of {len(records)} games to a named line")
    for match_type in ("exact", "transposition", "approximate", "none"):
        print(f"  {match_type:<14} {match_types[match_type]:>6}")
    print(f"Mean book depth: {np.mean(depths):.1f} plies (median {np.median(depths):.0f})")
    print(f"Matched ECO equals header ECO:     {100 * agree_code / matched:.1f}%")
    print(f"Same ECO group (e.g. B9x vs B9x):  {100 * agree_group / matched:.1f}%")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
