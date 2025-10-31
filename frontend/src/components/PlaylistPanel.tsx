import { useContext, useState } from "react";
import { PlaylistContext } from "../contexts/PlaylistContext";
import { useSocket } from "../contexts/SocketContext";
import { MDBIcon } from "mdb-react-ui-kit";

export default function PlaylistPanel() {
  const { current, playlists } = useContext(PlaylistContext);
  const { sendMessage } = useSocket();
  const [newPlaylistName, setNewPlaylistName] = useState<string>("");
  const [addInput, setAddInput] = useState<string>("");

  const playlist = playlists[current] || { name: current, tracks: [] };
  const cover = playlist.cover_url;

  function addTrack() {
    if (!addInput.trim()) return;
    sendMessage({ message: `/add ${addInput}` });
    setAddInput("");
  }
  function removeTrack(uriOrIndex: string) {
    sendMessage({ message: `/remove ${uriOrIndex}` });
  }
  function playTrack(index: number) {
    sendMessage({ message: `/play ${index}` });
  }
  function clearPlaylist() {
    sendMessage({ message: "/clear" });
  }
  function createPlaylist() {
    if (!newPlaylistName.trim()) return;
    sendMessage({ message: `/create ${newPlaylistName}` });
    setNewPlaylistName("");
  }
  function switchPlaylist(name: string) {
    sendMessage({ message: `/switch ${name}` });
  }

  return (
    <div className="p-2">
      <div className="d-flex align-items-center gap-2 mb-2">
        <h5 className="mb-0">Playlist</h5>
        <select
          className="form-select form-select-sm w-auto"
          value={current}
          onChange={(e) => switchPlaylist(e.target.value)}
        >
          {Object.keys(playlists).map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
        <input
          className="form-control form-control-sm w-auto"
          placeholder="New playlist name"
          value={newPlaylistName}
          onChange={(e) => setNewPlaylistName(e.target.value)}
        />
        <button className="btn btn-sm btn-outline-primary" onClick={createPlaylist}>
          <MDBIcon fas icon="plus" /> New
        </button>
      </div>

      <div className="mb-3">
        {cover ? (
          <img
            src={cover}
            alt="cover"
            style={{ width: "100%", maxWidth: "320px", borderRadius: "8px" }}
          />
        ) : (
          <div
            style={{
              width: "100%",
              maxWidth: "320px",
              height: "200px",
              borderRadius: "8px",
              background: "#f2f2f2",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontStyle: "italic",
            }}
          >
            No cover yet
          </div>
        )}
      </div>

      <div className="input-group mb-2">
        <input
          className="form-control"
          placeholder="Artist: Title"
          value={addInput}
          onChange={(e) => setAddInput(e.target.value)}
        />
        <button className="btn btn-primary" onClick={addTrack}>
          Add
        </button>
        <button className="btn btn-outline-danger" onClick={clearPlaylist}>
          Clear
        </button>
      </div>

      <ol className="list-group list-group-numbered">
        {(playlist.tracks || []).map((t, i) => (
          <li
            key={t.track_uri}
            className="list-group-item d-flex justify-content-between align-items-center"
          >
            <div className="me-2 flex-grow-1">
              <div className="fw-bold">{t.artist} – {t.title}</div>
              <div className="text-muted small">{t.album}</div>
              <div className="text-muted small">{t.track_uri}</div>
            </div>
            <div className="d-flex gap-1">
              <button
                className="btn btn-sm btn-success"
                title="Play/Preview on Spotify"
                onClick={() => playTrack(i + 1)}
              >
                <MDBIcon fas icon="play" />
              </button>
              <button
                className="btn btn-sm btn-outline-secondary"
                title="Remove"
                onClick={() => removeTrack(String(i + 1))}
              >
                <MDBIcon fas icon="trash" />
              </button>
            </div>
          </li>
        ))}
      </ol>

      <div className="mt-3">
        <details>
          <summary className="fw-bold" style={{ cursor: "pointer" }}>How to use</summary>
          <div className="mt-2">
            <h6>🎵 Natural Language (NEW!)</h6>
            <ul className="small">
              <li><strong>Add songs:</strong> "add one dance by drake", "play humble"</li>
              <li><strong>View playlist:</strong> "show my playlist", "what's in my playlist"</li>
              <li><strong>Remove songs:</strong> "remove track 3", "delete humble"</li>
              <li><strong>Get recommendations:</strong> "recommend 5 songs", "suggest similar music"</li>
              <li><strong>Select from recommendations:</strong> "add first two", "add all except drake"</li>
              <li><strong>Ask questions:</strong> "what artists are in my playlist", "how many songs by beatles"</li>
            </ul>
            
            <h6>Adding Songs (Structured)</h6>
            <ul className="small">
              <li><code>/add [title]</code> – add by title (disambiguation if multiple matches)</li>
              <li><code>/bulkadd [title]</code> – show all matches for a title</li>
              <li><code>/add [artist]: [title]</code> – colon-separated format</li>
              <li><code>/add [artist] - [title]</code> – dash-separated format</li>
            </ul>
            
            <h6>Recommendations</h6>
            <ul className="small">
              <li><code>/auto</code> – get automatic recommendations based on current playlist</li>
              <li>Or use natural language: "recommend 5 songs"</li>
            </ul>
            
            <h6>Managing Playlists</h6>
            <ul className="small">
              <li><code>/remove [index|uri]</code> – remove by number or track URI</li>
              <li><code>/view</code> – show current playlist</li>
              <li><code>/clear</code> – remove all tracks</li>
              <li><code>/create [name]</code> – create and switch to new playlist</li>
              <li><code>/switch [name]</code> – switch to a different playlist</li>
              <li><code>/list</code> – list all your playlists</li>
            </ul>
            
            <h6>Playback & Info</h6>
            <ul className="small">
              <li><code>/play [number]</code> – get Spotify embed for track in playlist</li>
              <li><code>/preview [artist/title]</code> – search and preview any track on Spotify</li>
              <li><code>/stats</code> – show playlist statistics (popularity, genres, etc.)</li>
            </ul>
            
            <h6>Questions & Search</h6>
            <ul className="small">
              <li><code>/ask [question]</code> – ask about tracks or artists</li>
              <li>Examples: "what album is hey jude by the beatles on", "how many songs by queen"</li>
              <li>Or use natural language without <code>/ask</code></li>
            </ul>
            
            <p className="text-muted small mb-0">
              <strong>💡 Tip:</strong> Try natural language first! The system handles typos, word order, and special characters.
            </p>
          </div>
        </details>
      </div>
    </div>
  );
}
