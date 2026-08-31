import { useState } from "react";
import { usePlayers } from "../hooks/usePlayers";
import { useTeams } from "../hooks/useTeams";

const POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"];

export function PlayersPage() {
  const [nameFilter, setNameFilter] = useState("");
  const [positionFilter, setPositionFilter] = useState("");
  const [teamFilter, setTeamFilter] = useState("");

  const { data: teams } = useTeams();
  const { data: players, isLoading, error } = usePlayers({
    name: nameFilter || undefined,
    position: positionFilter || undefined,
    team_abbr: teamFilter || undefined,
  });

  return (
    <div>
      <h1>Players</h1>

      <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
        <input
          type="text"
          placeholder="Search by name..."
          value={nameFilter}
          onChange={(e) => setNameFilter(e.target.value)}
        />

        <select value={positionFilter} onChange={(e) => setPositionFilter(e.target.value)}>
          <option value="">All positions</option>
          {POSITIONS.map((pos) => (
            <option key={pos} value={pos}>{pos}</option>
          ))}
        </select>

        <select value={teamFilter} onChange={(e) => setTeamFilter(e.target.value)}>
          <option value="">All teams</option>
          {teams?.map((team) => (
            <option key={team.team_abbr} value={team.team_abbr}>
              {team.team_name}
            </option>
          ))}
        </select>
      </div>

      {isLoading && <p>Loading players...</p>}
      {error && <p>Error loading players: {(error as Error).message}</p>}

      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Position</th>
            <th>Team</th>
          </tr>
        </thead>
        <tbody>
            {players?.map((player) => (
                <tr key={player.player_id}>
                <td>{player.full_name}</td>
                <td>{player.position ?? "—"}</td>
                <td>{player.current_team_abbr ?? "—"}</td>
                </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}