# Get data from Chess.com's API regarding Magnus Carlsen's matches.
#
# Uses the Chess.com Published-Data API (https://www.chess.com/news/view/published-data-api),
# which is public and read-only: no API key or login is needed.
#
# Usage:
#   python3 retrieve.py                  # fetch Magnus Carlsen's games
#   python3 retrieve.py some_username    # fetch another player's games

import json
import os
import sys
import time
import urllib.error
import urllib.request

USERNAME = "magnuscarlsen"
BASE_URL = "https://api.chess.com/pub/player/{username}/games/archives"
OUTPUT_DIR = "data"

# Chess.com asks clients to send a descriptive User-Agent, ideally with contact
# info, so they can reach you instead of blocking you. Set CHESSCOM_CONTACT to
# an email or URL to include it.
CONTACT = os.environ.get("CHESSCOM_CONTACT", "")
USER_AGENT = "ChessOpeningsAnalysis/1.0" + (f" (contact: {CONTACT})" if CONTACT else "")

MAX_RETRIES = 5


def get_json(url):
    """GET a URL and parse the JSON body, backing off on rate limits (429)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait = 2 ** attempt
                print(f"  HTTP {e.code} for {url}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Giving up on {url} after {MAX_RETRIES} attempts")


def fetch_all_games(username):
    """Return every game in the player's monthly archives, oldest first."""
    archives = get_json(BASE_URL.format(username=username.lower()))["archives"]
    print(f"Found {len(archives)} monthly archives for {username}")

    games = []
    # Requests are made one at a time on purpose: Chess.com rate-limits
    # parallel requests but allows serial access without limits.
    for i, archive_url in enumerate(archives, start=1):
        month_games = get_json(archive_url).get("games", [])
        games.extend(month_games)
        month = "/".join(archive_url.rsplit("/", 2)[-2:])
        print(f"  [{i}/{len(archives)}] {month}: {len(month_games)} games")

    return games


def main():
    username = sys.argv[1] if len(sys.argv) > 1 else USERNAME
    games = fetch_all_games(username)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{username.lower()}_games.json")
    with open(output_path, "w") as f:
        json.dump(games, f, indent=2)

    print(f"Saved {len(games)} games to {output_path}")


if __name__ == "__main__":
    main()
