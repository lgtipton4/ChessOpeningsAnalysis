# Group Magnus Carlsen's games into opening categories found by k-means
# clustering, instead of the ~3,000 hand-named openings.
#
# Named openings split hairs in some places (dozens of Sicilian sub-lines) and
# lump very different games together in others ("Queen's Pawn Game"). Clustering
# lets the categories come from the games themselves: games whose opening
# positions look alike end up in the same cluster, whatever they are called and
# whatever move order reached them.
#
# Features. Each game is described by the board after plies 2, 4, ..., 16
# (SNAPSHOT_PLIES). Each snapshot is encoded as 768 yes/no features, one per
# (piece type, color, square), the same encoding chess engines use. Snapshots
# taken at several plies capture both the early choices (1.e4 vs 1.d4) and the
# structure that follows. Because these are positions rather than move lists,
# transpositions get identical features.
#
# Clustering. The ~6,000 sparse features are reduced to SVD_COMPONENTS
# dimensions with truncated SVD (latent semantic analysis), then grouped with
# k-means into DEFAULT_K clusters. Silhouette scores for this data are low and
# nearly flat across k (about 0.11 at k=10 up to 0.13 at k=40): opening
# choices form a continuum rather than well-separated groups, so there is no
# "natural" k. DEFAULT_K is a readability choice; --choose-k prints the scores
# so other values can be compared.
#
# Each cluster is named after the most common named opening family among its
# games (from match_openings.py), or its top two families when neither holds a
# majority. Normalized mutual information (NMI) between clusters and named
# families says how closely the organic categories track the named ones.
#
# Usage:
#   python3 cluster_openings.py               # DEFAULT_K clusters
#   python3 cluster_openings.py --k 30        # another number of clusters
#   python3 cluster_openings.py --choose-k    # pick k by silhouette score

import argparse
import collections
import csv
import os

import chess
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

INPUT_PATH = os.path.join("data", "carlsen_games_openings.csv")
GAMES_OUTPUT_PATH = os.path.join("data", "carlsen_games_clustered.csv")
CLUSTERS_OUTPUT_PATH = os.path.join("data", "opening_clusters.csv")

SNAPSHOT_PLIES = (2, 4, 6, 8, 10, 12, 14, 16)
SVD_COMPONENTS = 50
DEFAULT_K = 20
K_CANDIDATES = range(10, 41, 5)
# Silhouette score is quadratic in the number of points, so it's estimated on
# a random sample.
SILHOUETTE_SAMPLE = 5000
# How many plies of each cluster's most common move order to show in its label.
TYPICAL_LINE_PLIES = 8
RANDOM_STATE = 0

FEATURES_PER_SNAPSHOT = 12 * 64


def position_features(moves):
    """Return the active feature indices for one game's snapshot positions.

    If a game ends before a snapshot ply, its final position is used.
    """
    board = chess.Board()
    snapshots = []
    for ply in range(1, SNAPSHOT_PLIES[-1] + 1):
        if ply <= len(moves):
            board.push_san(moves[ply - 1])
        if ply in SNAPSHOT_PLIES:
            snapshots.append(board.piece_map())

    indices = []
    for slot, pieces in enumerate(snapshots):
        offset = slot * FEATURES_PER_SNAPSHOT
        for square, piece in pieces.items():
            plane = (piece.piece_type - 1) + (6 if piece.color == chess.BLACK else 0)
            indices.append(offset + plane * 64 + square)
    return indices


def feature_matrix(games):
    """Return a sparse games x features matrix of 0/1 piece-square features."""
    rows, cols = [], []
    for row, game in enumerate(games):
        active = position_features(game["Moves"].split())
        rows.extend([row] * len(active))
        cols.extend(active)
    shape = (len(games), FEATURES_PER_SNAPSHOT * len(SNAPSHOT_PLIES))
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape)


def choose_k(embedding, candidates):
    """Return the k with the best silhouette score, printing each score."""
    rng = np.random.default_rng(RANDOM_STATE)
    sample = rng.choice(len(embedding), size=min(SILHOUETTE_SAMPLE, len(embedding)), replace=False)
    best_k, best_score = None, -1.0
    print("k   silhouette")
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(embedding)
        score = silhouette_score(embedding[sample], labels[sample])
        print(f"{k:<3} {score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def describe_clusters(games, labels):
    """Return one summary dict per cluster, largest cluster first."""
    members = collections.defaultdict(list)
    for game, label in zip(games, labels):
        members[label].append(game)

    summaries = []
    for label, group in members.items():
        families = collections.Counter(g["MatchedFamily"] for g in group).most_common(2)
        top_family, top_count = families[0]
        lines = collections.Counter(" ".join(g["Moves"].split()[:TYPICAL_LINE_PLIES]) for g in group)
        typical_line, line_count = lines.most_common(1)[0]
        summaries.append({
            "label": label,
            "games": len(group),
            "top_family": top_family,
            "top_family_share": top_count / len(group),
            "second_family": families[1][0] if len(families) > 1 else "",
            "typical_line": typical_line,
            "typical_line_share": line_count / len(group),
            "mean_book_depth": np.mean([int(g["BookDepth"]) for g in group]),
            "carlsen_white_share": sum(g["Color"] == "white" for g in group) / len(group),
        })
    summaries.sort(key=lambda s: -s["games"])

    # Name clusters after their main opening (or top two without a majority),
    # numbering repeats ("Sicilian Defense 1", "Sicilian Defense 2", ...) in
    # order of size.
    for s in summaries:
        s["base_name"] = s["top_family"]
        if s["top_family_share"] < 0.5 and s["second_family"]:
            s["base_name"] += " / " + s["second_family"]
    totals = collections.Counter(s["base_name"] for s in summaries)
    seen = collections.Counter()
    for s in summaries:
        seen[s["base_name"]] += 1
        s["name"] = s["base_name"] + (f" {seen[s['base_name']]}" if totals[s["base_name"]] > 1 else "")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="number of clusters")
    parser.add_argument("--choose-k", action="store_true",
                        help="pick k from K_CANDIDATES by silhouette score instead")
    parser.add_argument("--input", default=INPUT_PATH)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        games = list(csv.DictReader(f))

    features = feature_matrix(games)
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    embedding = svd.fit_transform(features)
    print(f"{features.shape[0]} games x {features.shape[1]} features -> {SVD_COMPONENTS} SVD components "
          f"({100 * svd.explained_variance_ratio_.sum():.0f}% of variance)")

    k = choose_k(embedding, K_CANDIDATES) if args.choose_k else args.k
    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(embedding)
    summaries = describe_clusters(games, labels)
    names = {s["label"]: s["name"] for s in summaries}
    # Renumber clusters by size so cluster 0 is the largest.
    numbers = {s["label"]: i for i, s in enumerate(summaries)}

    with open(GAMES_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[*games[0].keys(), "Cluster", "ClusterName"])
        writer.writeheader()
        for game, label in zip(games, labels):
            writer.writerow({**game, "Cluster": numbers[label], "ClusterName": names[label]})

    with open(CLUSTERS_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster", "Name", "Games", "TopFamily", "TopFamilyShare", "SecondFamily",
                         "TypicalLine", "TypicalLineShare", "MeanBookDepth", "CarlsenWhiteShare"])
        for s in summaries:
            writer.writerow([numbers[s["label"]], s["name"], s["games"], s["top_family"],
                             round(s["top_family_share"], 3), s["second_family"], s["typical_line"],
                             round(s["typical_line_share"], 3), round(s["mean_book_depth"], 1),
                             round(s["carlsen_white_share"], 3)])

    nmi = normalized_mutual_info_score([g["MatchedFamily"] for g in games], labels)
    print(f"\n{k} clusters (NMI with named opening families: {nmi:.2f}):")
    for s in summaries:
        print(f"  {numbers[s['label']]:>2}  {s['name']:<44} {s['games']:>5} games  "
              f"{100 * s['top_family_share']:>3.0f}% {s['top_family']:<28}  {s['typical_line']}")
    print(f"Wrote {GAMES_OUTPUT_PATH} and {CLUSTERS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
