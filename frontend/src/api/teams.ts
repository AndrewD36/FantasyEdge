import { apiFetch } from "./client";
import type { Team, RosterEntry } from "../types/team";
import type { Game } from "../types/game";

export function getTeams(): Promise<Team[]> {
  return apiFetch<Team[]>("/teams");
}

export function getTeam(teamAbbr: string): Promise<Team> {
  return apiFetch<Team>(`/teams/${teamAbbr}`);
}

export function getTeamRoster(
  teamAbbr: string,
  season: number,
  week?: number
): Promise<RosterEntry[]> {
  const params = new URLSearchParams({ season: String(season) });
  if (week !== undefined) params.set("week", String(week));
  return apiFetch<RosterEntry[]>(`/teams/${teamAbbr}/roster?${params}`);
}

export function getTeamSchedule(
  teamAbbr: string,
  season: number,
  week?: number
): Promise<Game[]> {
  const params = new URLSearchParams({ season: String(season) });
  if (week !== undefined) params.set("week", String(week));
  return apiFetch<Game[]>(`/teams/${teamAbbr}/schedule?${params}`);
}