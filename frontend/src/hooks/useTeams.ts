import { useQuery } from "@tanstack/react-query";
import { getTeams, getTeam, getTeamRoster, getTeamSchedule } from "../api/teams";

export function useTeams() {
  return useQuery({
    queryKey: ["teams"],
    queryFn: getTeams,
  });
}

export function useTeam(teamAbbr: string) {
  return useQuery({
    queryKey: ["team", teamAbbr],
    queryFn: () => getTeam(teamAbbr),
    enabled: !!teamAbbr,
  });
}

export function useTeamRoster(teamAbbr: string, season: number, week?: number) {
  return useQuery({
    queryKey: ["team-roster", teamAbbr, season, week],
    queryFn: () => getTeamRoster(teamAbbr, season, week),
    enabled: !!teamAbbr && !!season,
  });
}

export function useTeamSchedule(teamAbbr: string, season: number, week?: number) {
  return useQuery({
    queryKey: ["team-schedule", teamAbbr, season, week],
    queryFn: () => getTeamSchedule(teamAbbr, season, week),
    enabled: !!teamAbbr && !!season,
  });
}