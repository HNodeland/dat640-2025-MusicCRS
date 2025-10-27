# tools/ensure_reco_indexes.py
import os, sqlite3
DB = "data/mpd.sqlite"
con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("CREATE INDEX IF NOT EXISTS idx_pt_track ON playlist_tracks(track_uri)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_pt_pid   ON playlist_tracks(pid)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_uri ON tracks(track_uri)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_artist_title ON tracks(artist COLLATE NOCASE, title COLLATE NOCASE)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_pop ON tracks(popularity)")
con.commit()
con.close()
print("Recommendation indexes ensured.")
