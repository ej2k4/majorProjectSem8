import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav style={{ padding: "20px", backgroundColor: "#111827" }}>
      <Link to="/" style={{ color: "white" }}>
        Story + Cartoon Generator
      </Link>
    </nav>
  );
}

export default Navbar;