import { useTeams } from "../hooks/useTeams";

export function TeamsPage() {
  const { data: teams, isLoading, error } = useTeams();

  if (isLoading) return <p>Loading teams...</p>;
  if (error) return <p>Error loading teams: {(error as Error).message}</p>;

  return (
    <div>
      <h1>Teams</h1>
      <ul>
        {teams?.map((team) => (
          <li key={team.team_abbr}>
            {team.team_name} ({team.team_abbr})
            {team.conference && team.division && (
              <> — {team.conference} {team.division}</>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}