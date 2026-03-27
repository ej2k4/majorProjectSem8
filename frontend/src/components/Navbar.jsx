import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav style={{ padding: "20px", textAlign: "center" }}>
      <Link to="/" style={{ marginRight: "20px" }}>
         Story Generator
      </Link>

      <Link to="/sentence">
         Sentence Prediction
      </Link>

      <Link to="/game" style={{ marginLeft: "20px" }}>
         Gamified Learning
      </Link>
    </nav>
  );
}

export default Navbar;