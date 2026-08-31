// src/hooks/usePlayers.ts
import { useQuery } from "@tanstack/react-query";
import { getPlayers, type PlayerFilters } from "../api/players";

export function usePlayers(filters: PlayerFilters) {
  return useQuery({
    queryKey: ["players", filters],
    queryFn: () => getPlayers(filters),
  });
}