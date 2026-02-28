import { useState } from "react";
//import { predictASD } from "../services/api";

function ASDPredictor() {
  const [age, setAge] = useState("");
  const [score, setScore] = useState("");

  const handlePredict = async () => {
    const res = await predictASD({ age, score });
    alert("Prediction: " + res.data.result);
  };

  return (
    <div style={{ padding: "40px" }}>
      <h2>ASD Prediction</h2>

      <input
        placeholder="Age"
        onChange={(e) => setAge(e.target.value)}
      />

      <br /><br />

      <input
        placeholder="Screening Score"
        onChange={(e) => setScore(e.target.value)}
      />

      <br /><br />

      <button onClick={handlePredict}>Predict</button>
    </div>
  );
}

export default ASDPredictor;