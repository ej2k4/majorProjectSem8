import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav style={{ padding: "20px", backgroundColor: "#111827" }}>
      <Link to="/" style={{ color: "white", marginRight: "20px" }}>
        Story Generator
      </Link>
      <Link to="/asd" style={{ color: "white", marginRight: "20px" }}>
        ASD Prediction
      </Link>
      <Link to="/cartoon" style={{ color: "white" }}>
        Cartoon Generator
      </Link>
    </nav>
  );
}

export default Navbar;