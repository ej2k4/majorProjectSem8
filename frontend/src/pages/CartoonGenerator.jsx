import { useState } from "react";
import { generateCartoon } from "../services/api";

function CartoonGenerator() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState("");

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", image);

    const res = await generateCartoon(formData);
    setResult(res.data.image_url);
  };

  return (
    <div style={{ padding: "40px" }}>
      <h2>Cartoon Generator</h2>

      <input
        type="file"
        onChange={(e) => setImage(e.target.files[0])}
      />

      <br /><br />

      <button onClick={handleUpload}>Generate Cartoon</button>

      {result && <img src={result} width="300" alt="cartoon" />}
    </div>
  );
}

export default CartoonGenerator;