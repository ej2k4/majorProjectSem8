import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import StoryGenerator from "./pages/StoryGenerator";

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<StoryGenerator />} />
      </Routes>
    </Router>
  );
}

export default App;