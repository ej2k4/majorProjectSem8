import { useState } from "react";
import { generateStory } from "../services/api";

function StoryGenerator() {
  const [scenario, setScenario] = useState("doctor_visit");
  const [name, setName] = useState("");
  const [story, setStory] = useState("");

  const handleGenerate = async () => {
    if (!name) {
      alert("Enter child name");
      return;
    }

    const res = await generateStory({ scenario, name });
    setStory(res.data.story);
  };

  return (
    <div style={{ padding: "40px" }}>
      <h2>AI Comfort Story Generator</h2>

      <select onChange={(e) => setScenario(e.target.value)}>
        <option value="doctor_visit">Doctor Visit</option>
        <option value="dentist">Dentist</option>
        <option value="haircut">Haircut</option>
      </select>

      <br /><br />

      <input
        placeholder="Child Name"
        onChange={(e) => setName(e.target.value)}
      />

      <br /><br />

      <button onClick={handleGenerate}>Generate</button>

      <p style={{ marginTop: "20px" }}>{story}</p>
    </div>
  );
}

export default StoryGenerator;