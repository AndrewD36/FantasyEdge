export interface Game {
  game_id: string;
  season: number;
  week: number;
  game_type: string;
  gameday: string | null; // ISO date string
  weekday: string | null;
  gametime: string | null;

  away_team: string;
  home_team: string;
  away_score: number | null;
  home_score: number | null;
  result: number | null;
  total: number | null;
  overtime: boolean;
  div_game: boolean;

  roof: string | null;
  surface: string | null;
  temp: number | null;
  wind: number | null;

  away_qb_id: string | null;
  home_qb_id: string | null;
  away_qb_name: string | null;
  home_qb_name: string | null;
  away_coach: string | null;
  home_coach: string | null;
  referee: string | null;
  stadium: string | null;

  spread_line: number | null;
  total_line: number | null;
}