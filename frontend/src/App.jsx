import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import StoryGenerator from "./pages/StoryGenerator";
import SentencePrediction from "./pages/SentencePrediction";

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<StoryGenerator />} />
        <Route path="/sentence" element={<SentencePrediction />} />
      </Routes>
    </Router>
  );
}

export default App;