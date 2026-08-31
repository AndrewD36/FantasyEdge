// src/api/players.ts
import { apiFetch } from "./client";
import type { Player } from "../types/player";

export interface PlayerFilters {
  name?: string;
  position?: string;
  team_abbr?: string;
}

export function getPlayers(filters: PlayerFilters): Promise<Player[]> {
  const params = new URLSearchParams();
  if (filters.name) params.set("name", filters.name);
  if (filters.position) params.set("position", filters.position);
  if (filters.team_abbr) params.set("team_abbr", filters.team_abbr);

  const query = params.toString();
  return apiFetch<Player[]>(`/players${query ? `?${query}` : ""}`);
}