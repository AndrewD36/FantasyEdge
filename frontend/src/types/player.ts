export interface Player {
  player_id: string;
  full_name: string;
  first_name: string | null;
  last_name: string | null;
  position: string | null;
  status: string | null;
  current_team_abbr: string | null;
}