import { Link, Outlet } from "react-router-dom";

function App() {
  return (
    <div>
      <nav>
        <Link to="/">Home</Link> | <Link to="/players">Players</Link>
      </nav>
      <Outlet />
    </div>
  );
}

export default App;