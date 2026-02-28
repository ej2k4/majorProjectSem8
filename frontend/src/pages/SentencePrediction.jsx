import { useState } from "react";
import { predictSentence } from "../services/api";

function SentencePrediction() {
  const [inputText, setInputText] = useState("");
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!inputText) {
      alert("Please enter a sentence");
      return;
    }

    try {
      setLoading(true);
      const res = await predictSentence({ text: inputText });
      setPrediction(res.data.prediction);
    } catch (error) {
      console.error(error);
      alert("Prediction failed ❌");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h2>Sentence Prediction</h2>

      <input
        placeholder="Enter sentence"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
      />

      <br /><br />

      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Predicting..." : "Predict Sentence"}
      </button>

      {prediction && (
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
            textAlign: "left"
          }}
        >
          <h3
            style={{
              fontSize: "20px",
              marginBottom: "15px",
              fontWeight: "700"
            }}
          >
            Predicted Sentence
          </h3>

          {prediction}
        </div>
      )}
    </div>
  );
}

export default SentencePrediction;