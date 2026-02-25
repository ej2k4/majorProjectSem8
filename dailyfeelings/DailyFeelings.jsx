import React, { useState, useEffect } from "react";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Gaegu:wght@400;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #8faf7e;
    font-family: 'Gaegu', cursive;
    min-height: 100vh;
    overflow: hidden;
  }

  .app {
    width: 100vw; height: 100vh;
    display: flex; align-items: center; justify-content: center;
    position: relative;
  }

  .splash {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 40px;
    animation: fadeIn 0.8s ease;
  }

  .splash-text {
    font-family: 'Gaegu', cursive;
    font-size: clamp(1.8rem, 4vw, 2.6rem);
    font-weight: 700;
    color: rgba(255,255,255,0.92);
    display: flex; align-items: center; gap: 10px;
    letter-spacing: 0.5px;
  }

  .feel-word { color: #4a6e3a; font-size: 1.1em; display: inline-block; }

  .cursor {
    display: inline-block; width: 2px; height: 1.1em;
    background: #4a6e3a; margin-left: 2px; vertical-align: middle;
    animation: blink 1s step-end infinite;
  }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

  .btn-write {
    background: white; border: none; border-radius: 50px; padding: 14px 44px;
    font-family: 'Gaegu', cursive; font-size: 1.2rem; font-weight: 700; color: #3d3d3d;
    cursor: pointer; box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .btn-write:hover { transform: scale(1.06); box-shadow: 0 8px 28px rgba(0,0,0,0.18); }

  .feelings-page {
    width: min(680px, 96vw); display: flex; flex-direction: column;
    align-items: center; gap: 24px;
    animation: slideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1);
  }
  @keyframes slideUp { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }

  .page-title {
    font-family: 'Gaegu', cursive; font-size: clamp(1.6rem, 4vw, 2.2rem);
    font-weight: 700; color: rgba(255,255,255,0.95); text-align: center;
  }

  .emotions-row { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }

  .emotion-btn {
    background: #6b8f5a; border: 2px solid transparent; border-radius: 14px;
    padding: 10px 8px 8px; width: 76px;
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    cursor: pointer; transition: transform 0.2s, background 0.2s, border-color 0.2s, box-shadow 0.2s;
  }
  .emotion-btn:hover { transform: translateY(-4px) scale(1.05); box-shadow: 0 8px 20px rgba(0,0,0,0.2); }
  .emotion-btn.selected {
    background: #5a7a4a; border-color: rgba(255,255,255,0.6);
    box-shadow: 0 0 0 3px rgba(255,255,255,0.25), 0 6px 20px rgba(0,0,0,0.2);
    transform: translateY(-4px) scale(1.08);
  }

  .emotion-face { width: 46px; height: 46px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 28px; }
  .emotion-label { font-family: 'Gaegu', cursive; font-size: 0.8rem; font-weight: 700; color: rgba(255,255,255,0.9); text-align: center; }

  .textareas-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; width: 100%; }
  .textarea-group { display: flex; flex-direction: column; gap: 8px; }
  .textarea-label { font-family: 'Gaegu', cursive; font-size: 1rem; font-weight: 700; color: rgba(255,255,255,0.9); }

  .feeling-textarea {
    background: rgba(255,255,255,0.92); border: none; border-radius: 12px; padding: 14px;
    font-family: 'Gaegu', cursive; font-size: 1rem; color: #3d3d3d;
    resize: none; height: 100px; outline: none;
    transition: box-shadow 0.2s; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
  }
  .feeling-textarea:focus { box-shadow: 0 0 0 3px rgba(255,255,255,0.4), 0 4px 16px rgba(0,0,0,0.1); }
  .feeling-textarea::placeholder { color: #b0b0b0; }

  .btn-save {
    background: white; border: none; border-radius: 50px; padding: 13px 48px;
    font-family: 'Gaegu', cursive; font-size: 1.15rem; font-weight: 700; color: #3d3d3d;
    cursor: pointer; box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    transition: transform 0.2s, box-shadow 0.2s; margin-top: 4px;
  }
  .btn-save:hover { transform: scale(1.05); }

  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.35); backdrop-filter: blur(4px);
    display: flex; align-items: center; justify-content: center; z-index: 200;
  }
  .modal {
    background: white; border-radius: 24px; padding: 36px 32px; width: min(380px, 92vw);
    display: flex; flex-direction: column; align-items: center; gap: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.2);
  }
  .modal h3 { font-family:'Gaegu',cursive; font-size:1.5rem; font-weight:700; color:#3d3d3d; text-align:center; }
  .modal p { font-family:'Gaegu',cursive; font-size:0.95rem; color:#888; text-align:center; line-height:1.5; }
  .format-btns { display:flex; gap:12px; width:100%; }
  .btn-format {
    flex:1; padding:14px; border-radius:14px; border:2px solid #e0e0e0;
    background:white; font-family:'Gaegu',cursive; font-size:1.1rem; font-weight:700;
    color:#555; cursor:pointer; transition:all 0.2s; text-align:center;
  }
  .btn-format:hover { border-color:#8faf7e; background:#f0f7ec; color:#4a6e3a; transform:translateY(-2px); }
  .btn-cancel { background:transparent; border:none; font-family:'Gaegu',cursive; font-size:0.95rem; color:#bbb; cursor:pointer; }

  .toast {
    position: fixed; bottom: 32px; left: 50%; transform: translateX(-50%);
    background: rgba(50,70,40,0.92); color: white; padding: 12px 28px; border-radius: 50px;
    font-family: 'Gaegu', cursive; font-size: 1rem; font-weight: 700;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2); z-index: 100;
  }

  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
`;

const EMOTIONS = [
  { name: "Happy",       emoji: "😊", bg: "#c8e6a0" },
  { name: "Loved",       emoji: "🥰", bg: "#f7b8c8" },
  { name: "Confident",   emoji: "😎", bg: "#fde68a" },
  { name: "Playful",     emoji: "😄", bg: "#fcd34d" },
  { name: "Embarrassed", emoji: "😳", bg: "#a7f3d0" },
  { name: "Angry",       emoji: "😠", bg: "#fca5a5" },
  { name: "Scared",      emoji: "😨", bg: "#c4b5fd" },
  { name: "Sad",         emoji: "😢", bg: "#bfdbfe" },
];

const WORDS = ["grateful", "happy", "hopeful", "loved", "calm", "brave", "joyful"];

export default function DailyFeelings() {
  const [screen, setScreen] = useState("splash");
  const [typedWord, setTypedWord] = useState("");
  const [wordIdx, setWordIdx] = useState(0);
  const [charIdx, setCharIdx] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [selected, setSelected] = useState(null);
  const [why, setWhy] = useState("");
  const [wish, setWish] = useState("");
  const [toast, setToast] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [pendingEntry, setPendingEntry] = useState(null);

  const triggerDownload = (blob, name) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = name; a.click();
    URL.revokeObjectURL(url);
  };

  const downloadFile = (format) => {
    const entry = pendingEntry;
    const filename = "daily-feelings-" + entry.date;
    if (format === "json") {
      triggerDownload(new Blob([JSON.stringify([entry], null, 2)], { type: "application/json" }), filename + ".json");
    } else {
      const header = "Date,Emotion,What Made Me Feel That Way,What I Wish For Tomorrow\n";
      const row = '"' + entry.date + '","' + entry.emotion + '","' + entry.why.replace(/"/g, '""') + '","' + entry.wish.replace(/"/g, '""') + '"\n';
      triggerDownload(new Blob([header + row], { type: "text/csv" }), filename + ".csv");
    }
    setShowModal(false);
    setToast(true);
    setTimeout(() => { setToast(false); setScreen("splash"); setSelected(null); setWhy(""); setWish(""); }, 2200);
  };

 const handleSave = () => {
  const entry = {
    date: new Date().toISOString().split("T")[0],
    emotion: selected || "Not selected",
    why, wish,
  };

  // Send to backend to save in project folder
  fetch("http://localhost:5000/save-feeling", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  });

  setToast(true);
  setTimeout(() => { setToast(false); setScreen("splash"); setSelected(null); setWhy(""); setWish(""); }, 2200);
};

  useEffect(() => {
    if (screen !== "splash") return;
    const word = WORDS[wordIdx];
    let timeout;
    if (!deleting && charIdx < word.length) {
      timeout = setTimeout(() => { setTypedWord(word.slice(0, charIdx + 1)); setCharIdx(c => c + 1); }, 100);
    } else if (!deleting && charIdx === word.length) {
      timeout = setTimeout(() => setDeleting(true), 1400);
    } else if (deleting && charIdx > 0) {
      timeout = setTimeout(() => { setTypedWord(word.slice(0, charIdx - 1)); setCharIdx(c => c - 1); }, 60);
    } else if (deleting && charIdx === 0) {
      setDeleting(false);
      setWordIdx(i => (i + 1) % WORDS.length);
    }
    return () => clearTimeout(timeout);
  }, [charIdx, deleting, wordIdx, screen]);

  return (
    <React.Fragment>
      <style>{styles}</style>
      <div className="app">

        {screen === "splash" && (
          <div className="splash">
            <div className="splash-text">
              Today, I feel&nbsp;
              <span className="feel-word">
                {typedWord}<span className="cursor" />
              </span>
            </div>
            <button className="btn-write" onClick={() => setScreen("feelings")}>
              Write feelings
            </button>
          </div>
        )}

        {screen === "feelings" && (
          <div className="feelings-page">
            <div className="page-title">How I feel today ?</div>
            <div className="emotions-row">
              {EMOTIONS.map(e => (
                <button
                  key={e.name}
                  className={"emotion-btn" + (selected === e.name ? " selected" : "")}
                  onClick={() => setSelected(e.name)}
                >
                  <div className="emotion-face" style={{ background: e.bg }}>{e.emoji}</div>
                  <div className="emotion-label">{e.name}</div>
                </button>
              ))}
            </div>
            <div className="textareas-row">
              <div className="textarea-group">
                <div className="textarea-label">What made me feel that way ?</div>
                <textarea className="feeling-textarea" placeholder="Write here..." value={why} onChange={ev => setWhy(ev.target.value)} />
              </div>
              <div className="textarea-group">
                <div className="textarea-label">What I wish for myself tomorrow ?</div>
                <textarea className="feeling-textarea" placeholder="Write here..." value={wish} onChange={ev => setWish(ev.target.value)} />
              </div>
            </div>
            <button className="btn-save" onClick={handleSave}>Save for today</button>
          </div>
        )}

        {showModal && (
          <div className="modal-overlay" onClick={() => setShowModal(false)}>
            <div className="modal" onClick={e => e.stopPropagation()}>
              <div style={{ fontSize: "2.5rem" }}>💾</div>
              <h3>Save your feelings</h3>
              <p>Choose a format to download your entry for <strong>{pendingEntry && pendingEntry.date}</strong></p>
              <div className="format-btns">
                <button className="btn-format" onClick={() => downloadFile("csv")}>
                  📊 CSV
                  <div style={{ fontSize: "0.75rem", fontWeight: 400, color: "#aaa", marginTop: 4 }}>Excel / Sheets</div>
                </button>
                <button className="btn-format" onClick={() => downloadFile("json")}>
                  🗂️ JSON
                  <div style={{ fontSize: "0.75rem", fontWeight: 400, color: "#aaa", marginTop: 4 }}>Developers</div>
                </button>
              </div>
              <button className="btn-cancel" onClick={() => setShowModal(false)}>Cancel</button>
            </div>
          </div>
        )}

        {toast && <div className="toast">✨ Feelings saved and downloaded!</div>}
      </div>
    </React.Fragment>
  );
}
