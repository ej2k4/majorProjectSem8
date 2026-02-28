import { useState } from "react";
import { generateFull } from "../services/api";

function StoryGenerator() {
  const [scenario, setScenario] = useState("doctor_visit");
  const [name, setName] = useState("");
  const [emotion, setEmotion] = useState("");
  const [story, setStory] = useState("");
  const [image, setImage] = useState("");
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!name || !emotion) {
      alert("Enter child name and emotion");
      return;
    }

    try {
      setLoading(true);

      const res = await generateFull({
        scenario,
        name,
        emotion,
      });

      setStory(res.data.story);
      setImage(`data:image/png;base64,${res.data.image}`);

    } catch (err) {
      console.error(err);
      alert("Generation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h2>AI Comfort Story + Cartoon Generator</h2>

      <select
        value={scenario}
        onChange={(e) => setScenario(e.target.value)}
      >
        <option value="doctor_visit">Doctor Visit</option>
        <option value="dentist">Dentist</option>
        <option value="haircut">Haircut</option>
      </select>

      <br /><br />

      <input
        placeholder="Child Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />

      <br /><br />

      <input
        placeholder="Emotion (e.g., nervous, scared, excited)"
        value={emotion}
        onChange={(e) => setEmotion(e.target.value)}
      />

      <br /><br />

      <button onClick={handleGenerate} disabled={loading}>
        {loading ? "Generating..." : "Generate"}
      </button>

      {story && (
      <div
        style={{
          marginTop: "30px",
          width: "500px",
          maxWidth: "75%",
          marginLeft: "auto",
          marginRight: "auto",
          padding: "25px",
          background: "linear-gradient(135deg, #FFDEEB, #D0BFFF)",
          borderRadius: "20px",
          boxShadow: "0 8px 25px rgba(0,0,0,0.15)",
          fontSize: "18px",
          lineHeight: "1.8",
          fontWeight: "500",
          color: "#333",
          textAlign: "left",
          whiteSpace: "pre-line"
        }}
      >
        <h3 style={{ fontSize: "20px", marginBottom: "15px", fontWeight: "700" }}>
          📖 Your Comfort Story
        </h3>
        {story}
      </div>
    )}

      {image && (
  <div style={{ marginTop: "30px", textAlign: "center" }}>
    <img
      src={image}
      alt="Generated Cartoon"
      style={{
        width: "500px",
        maxWidth: "75%",
        borderRadius: "20px",
        boxShadow: "0 8px 25px rgba(0,0,0,0.2)"
      }}
    />
  </div>
)}
    </div>
  );
}

export default StoryGenerator;