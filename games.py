# Load Magnus Carlsen's games, and the named opening lines, as move sequences.
#
# The ECO code in a game's header doesn't say how closely the players followed
# an opening. Comparing openings by their actual moves (match_openings.py,
# cluster_openings.py) needs the moves themselves, so this module reads both
# game sources and the Lichess opening tables into lists of SAN moves. Every move is replayed on a board with
# python-chess so the SAN is written the same way in every source (check
# marks, disambiguation, castling notation).
#
# Only the first MAX_PLIES half-moves of each game are kept: the longest named
# line in the Lichess tables is 36 plies, so later moves never affect matching.

import csv
import datetime
import glob
import json
import os
import re

import chess

PGN_PATH = os.path.join("data", "Carlsen professional games.pgn")
JSON_PATH = os.path.join("data", "carlsen online chess.com games.json")
# Opening names come from the Lichess opening tables (CC0, from
# https://github.com/lichess-org/chess-openings).
ECO_DIR = os.path.join("data", "eco")

# PGN: matches "Carlsen,M" and "Carlsen,Magnus" but not other Carlsens (e.g. "Carlsen,H").
PGN_PLAYER_PATTERN = re.compile(r"^Carlsen,\s*M")
CHESSCOM_USERNAME = "magnuscarlsen"

TAG_PATTERN = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$', re.MULTILINE)

MAX_PLIES = 36

# Everything in PGN movetext that isn't a move: comments, variations, NAGs,
# move numbers ("12." / "12..."), and results.
MOVETEXT_NOISE = re.compile(r"\{[^}]*\}|\([^()]*\)|\$\d+|\d+\.+|1-0|0-1|1/2-1/2|\*")


def parse_moves(movetext, max_plies=MAX_PLIES):
    """Return the first max_plies moves of a PGN movetext as normalized SAN.

    Stops early at the first token that isn't a legal move, so a corrupt game
    keeps whatever prefix was readable.
    """
    board = chess.Board()
    moves = []
    for token in MOVETEXT_NOISE.sub(" ", movetext).split():
        if len(moves) >= max_plies:
            break
        try:
            move = board.parse_san(token)
        except ValueError:
            break
        moves.append(board.san(move))
        board.push(move)
    return moves


def opening_family(name):
    """'Sicilian Defense: Najdorf Variation' -> 'Sicilian Defense'.

    A few Lichess names put a setup before the colon ("Queen's Indian Defense,
    with e3: ..."); those belong to the plain family too.
    """
    return name.split(":")[0].split(", with ")[0]


def position_keys(moves):
    """Return a key for the position after each move, so transpositions compare equal.

    The key is the EPD (pieces, side to move, castling rights, en passant
    square), which ignores the move counters.
    """
    board = chess.Board()
    keys = []
    for move in moves:
        board.push_san(move)
        keys.append(board.epd())
    return keys


def load_opening_lines(eco_dir=ECO_DIR):
    """Return [{"eco", "name", "moves"}] for every named line in the Lichess tables."""
    lines = []
    for path in sorted(glob.glob(os.path.join(eco_dir, "*.tsv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                moves = parse_moves(row["pgn"], max_plies=10_000)
                lines.append({"eco": row["eco"], "name": row["name"], "moves": moves})
    return lines


def read_pgn_records(pgn_path=PGN_PATH):
    """Yield a record dict for each game in the PGN file.

    Keys: source, game_id, year, color, time_class, eco, moves. Games that
    start from a custom position (a FEN tag) are skipped.
    """
    headers, movetext, index = {}, [], 0

    def finish():
        if PGN_PLAYER_PATTERN.match(headers.get("White", "")):
            color = "white"
        elif PGN_PLAYER_PATTERN.match(headers.get("Black", "")):
            color = "black"
        else:
            color = None
        year = headers.get("Date", "")[:4]
        return {
            "source": "PGN",
            "game_id": f"pgn-{index}",
            "year": int(year) if year.isdigit() else None,
            "color": color,
            # The PGN mixes classical, rapid and blitz events without a
            # TimeControl tag, so its time class is unknown.
            "time_class": "",
            "eco": headers.get("ECO", "").strip().upper(),
            "moves": parse_moves(" ".join(movetext)),
        }

    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            match = TAG_PATTERN.match(line)
            if not match:
                movetext.append(line)
                continue
            tag, value = match.groups()
            # Every game starts with an Event tag.
            if tag == "Event" and headers:
                if "FEN" not in headers:
                    yield finish()
                headers, movetext, index = {}, [], index + 1
            headers[tag] = value
    if headers and "FEN" not in headers:
        yield finish()


def read_chesscom_records(json_path=JSON_PATH):
    """Yield a record dict (same keys as read_pgn_records) per standard Chess.com game."""
    with open(json_path, encoding="utf-8") as f:
        games = json.load(f)

    for game in games:
        if game.get("rules") != "chess" or game.get("initial_setup", chess.STARTING_FEN) != chess.STARTING_FEN:
            continue
        pgn = game.get("pgn", "")
        headers = dict(TAG_PATTERN.findall(pgn))
        if game["white"]["username"].lower() == CHESSCOM_USERNAME:
            color = "white"
        elif game["black"]["username"].lower() == CHESSCOM_USERNAME:
            color = "black"
        else:
            color = None
        year = datetime.datetime.fromtimestamp(game["end_time"], datetime.timezone.utc).year
        yield {
            "source": "Chess.com",
            "game_id": game.get("uuid") or game.get("url", ""),
            "year": year,
            "color": color,
            "time_class": game.get("time_class", ""),
            "eco": headers.get("ECO", "").strip().upper(),
            "moves": parse_moves(TAG_PATTERN.sub("", pgn)),
        }


def read_all_records():
    """Return every game from both sources as a list of record dicts."""
    return list(read_pgn_records()) + list(read_chesscom_records())
