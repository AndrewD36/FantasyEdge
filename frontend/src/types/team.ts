export interface Team {
  team_abbr: string;
  team_name: string;
  conference: string | null;
  division: string | null;
}

export interface RosterEntry {
  player_id: string;
  full_name: string;
  position: string | null;
  week: number;
}