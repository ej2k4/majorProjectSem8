import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import StoryGenerator from "./pages/StoryGenerator";
import ASDPredictor from "./pages/ASDPredictor";
import CartoonGenerator from "./pages/CartoonGenerator";

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<StoryGenerator />} />
        <Route path="/asd" element={<ASDPredictor />} />
        <Route path="/cartoon" element={<CartoonGenerator />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;