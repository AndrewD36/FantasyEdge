"""
Monte Carlo simulation engine for fantasy draft availability prediction.

This is the core intelligence of the system - it runs thousands of
simulated drafts to predict player availability and inform strategy.
"""

import asyncio
import logging
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, Counter
import numpy as np

from ..datamodels.draft_state import DraftState, TeamRoster
from ..datamodels.player import Player, PlayerPosition
from ..datamodels.simulation import (
    SimulationParameters, SimulationResult, PlayerAvailability,
    SimulationStrategy
)
from .draft_behavior import DraftBehaviorSimulator
from ..utils.snake_draft import SnakeDraftCalculator


logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for draft availability prediction.
    
    The core algorithm:
    1. For each simulation iteration:
       - Simulate other teams' picks until user's next turn
       - Track which players get drafted
    2. Aggregate results across all iterations
    3. Calculate availability probabilities and insights
    
    Design principles:
    - Deterministic given same seed (for testing)
    - Configurable behavior models
    - Statistically rigorous result calculation
    - Performance optimized for real-time use
    """

    def __init__(self, players_db: Dict[str, Player], thread_pool_size: int = 4, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            players_db: Dictionary of all available players
            thread_pool_size: Number of threads for parallel simulation
            random_seed: Optional seed for reproducible results
        """
        self.players_db = players_db
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self.behavior_simulator = DraftBehaviorSimulator(players_db)
        self.draft_calculator = SnakeDraftCalculator()

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        logger.info(f'Initialized Monte Carlo simulation with {len(players_db)} players.')

    async def run_simulation(self, draft_state: DraftState, parameters: SimulationParameters, target_players: Optional[List[str]] = None) -> SimulationResult:
        """
        Run complete Monte Carlo simulation for draft availability.
        
        This is the main entry point that orchestrates the entire simulation process.
        
        Args:
            draft_state: Current state of the draft
            parameters: Simulation configuration
            target_players: Optional list of specific players to analyze
        
        Returns:
            Complete simulation results with availability predictions
        """
        start_time = time.time()
        simulation_id = f'sim_{int(start_time)}_{random.randint(1000, 9999)}'

        logger.info(f'Starting simulation {simulation_id} with {parameters.num_simulations} iterations.')

        try:
            if target_players:
                players_to_analyze = [self.players_db[pid] for pid in target_players if pid in self.players_db and pid not in draft_state.drafted_players]
            else:
                players_to_analyze = self._get_relevant_players(draft_state, limit=100)

            logger.info(f'Analyzing {len(players_to_analyze)} players.')

            picks_until_user = self._calculate_picks_until_user_turn(draft_state)

            if picks_until_user <= 0:
                return self._create_immediate_availability_result(simulation_id, draft_state, parameters, players_to_analyze)
            
            simulation_results = await self._run_parallel_simulations(draft_state, parameters, players_to_analyze, picks_until_user)

            player_availabilities = self._aggregate_simulation_results(simulation_results, players_to_analyze, parameters.num_simulations)

            insights = self._generate_draft_insights(player_availabilities, draft_state, parameters)

            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000

            result = SimulationResult(
                simulation_id=simulation_id,
                draft_id=draft_state.draft_id,
                parameters=parameters,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.fromtimestamp(end_time),
                execution_time_ms=execution_time_ms,
                player_availabilities=player_availabilities,
                position_scarcity=insights['position_scarcity'],
                likely_runs=insights['likely_runs'],
                value_opportunities=insights['value_opportunities'],
                converged=True,  # TODO: Implement convergence detection
                convergence_iteration=None,
                confidence_interval=0.95,  # TODO: Calculate actual CI
                current_pick=draft_state.current_pick,
                picks_simulated=picks_until_user,
                user_team_id=draft_state.user_team_id
            )

            logger.info(f'Simulation {simulation_id} completed in {execution_time_ms}ms.')
            return result
        
        except Exception as e:
            logger.error(f'Simulation {simulation_id} falied: {e}')
            raise

    def _get_relevant_players(self, draft_state: DraftState, limit: int = 100) -> List[Player]:
        available_players = [player for player in self.players_db.values() if player.id not in draft_state.drafted_players]
        available_players.sort(key=lambda p: p.adp)

        return available_players[:limit]

    def _calculate_picks_until_user_turn(self, draft_state: DraftState) -> int:
        if not draft_state.user_team_id:
            return 10
        
        return self.draft_calculator.picks_until_team_turn(current_pick=draft_state.current_pick,
                                                           team_count=draft_state.team_count,
                                                           target_team_index=int(draft_state.user_team_id) - 1,
                                                           is_snake_draft=draft_state.draft_type == "snake")

    async def _run_parallel_simulations(self, 
                                        draft_state: DraftState, 
                                        parameters: SimulationParameters, 
                                        players_to_analyze: List[Player], 
                                        picks_to_simulate: int) -> List[Dict[str, Set[str]]]:
        simulations_per_batch = max(1, parameters.num_simulations // self.thread_pool._max_workers)
        tasks = []

        loop = asyncio.get_event_loop()

        for i in range(0, parameters.num_simulations, simulations_per_batch):
            batch_size = min(simulations_per_batch, parameters.num_simulations - i)
            
            task = loop.run_in_executor(self.thread_pool,
                                        self._run_simulation_batch,
                                        draft_state,
                                        parameters,
                                        players_to_analyze,
                                        picks_to_simulate,
                                        batch_size,
                                        i ) # Seed offset for randomness
            
            tasks.append(task)

        batch_results = await asyncio.gather(*tasks)

        all_results = []
        for batch in batch_results:
            all_results.extend(batch)

        return all_results

    def _run_simulation_batch(self, 
                            draft_state: DraftState,
                            parameters: SimulationParameters,
                            players_to_analyze: List[Player],
                            picks_to_simulate: int,
                            batch_size: int,
                            seed_offset: int) -> List[Dict[str, Set[str]]]:
        batch_results = []

        local_random = random.Random(seed_offset)

        for i in range(batch_size):
            result = self._simulate_single_scenario(draft_state, parameters, players_to_analyze, picks_to_simulate, local_random)
            batch_results.append(result)

        return batch_results

    def _simulate_single_scenario(self, 
                                draft_state: DraftState,
                                parameters: SimulationParameters,
                                players_to_analyze: List[Player],
                                picks_to_simulate: int,
                                rng: random.Random) -> Dict[str, Set[str]]:
        sim_state = self._create_simulation_state(draft_state)
        available_players = [p for p in self.players_db.values() if p.id not in sim_state['drafted_players']]

        picks_made = defaultdict(set)

        for pick_offset in range(picks_to_simulate):
            if not available_players:
                break

            current_pick_num = sim_state['current_pick'] + pick_offset
            picking_team = self.draft_calculator.get_picking_team(pick_number=current_pick_num,
                                                                team_count=draft_state.team_count,
                                                                is_snake_draft=draft_state.draft_type == "snake")
            
            selected_player = self.behavior_simulator.simulate_team_pick(team_id=picking_team,
                                                                        available_players=available_players,
                                                                        draft_context={'current_pick': current_pick_num,
                                                                                        'team_roster': sim_state['teams'][picking_team],
                                                                                        'drafted_players': sim_state['drafted_players'],
                                                                                        parameters: parameters},
                                                                        rng=rng)
            
            if selected_player:
                # Record the pick
                    picks_made[picking_team].add(selected_player.id)
                    sim_state['drafted_players'].add(selected_player.id)
                    sim_state['teams'][picking_team].append(selected_player.id)
                    available_players.remove(selected_player)
        
        return dict(picks_made)

    def _create_simulation_state(self, draft_state: DraftState) -> Dict:
        return {'current_pick': draft_state.current_pick,
                'drafted_players': set(draft_state.drafted_players),
                'teams': {team_id: list(roster.players) for team_id, roster in draft_state.teams.items()}}

    def _aggregate_simulation_results(self, 
                                    simulation_results: List[Dict[str, Set[str]]],
                                    players_analyzed: List[Player],
                                    total_simulations: int) -> Dict[str, PlayerAvailability]:
        player_draft_counts = Counter()
        player_draft_teams = defaultdict(list)
        player_pick_numbers = defaultdict(list)

        for sim_idx, sim_result in enumerate(simulation_results):
            for team_id, drafted_players in sim_result.items():
                for pick_offset, player_id in enumerate(drafted_players):
                    player_draft_counts[player_id] += 1
                    player_draft_teams[player_id].append(team_id)

                    player_pick_numbers[player_id].append(pick_offset + 1)

        availabilities = {}
        for player in players_analyzed:
            player_id = player.id
            times_drafted = player_draft_counts[player_id]
            times_available = total_simulations - times_drafted

            availability_prob = times_available / total_simulations

            pick_numbers = player_pick_numbers[player_id] or [999]
            avg_pick = np.mean(pick_numbers) if pick_numbers else 999
            earliest_pick = min(pick_numbers) if pick_numbers else 999
            latest_pick = max(pick_numbers) if pick_numbers else 999

            # Generate pick distribution (simplified)
            pick_distribution = {}
            if pick_numbers:
                pick_counter = Counter(pick_numbers)
                for pick_num, count in pick_counter.items():
                    pick_distribution[pick_num] = count / len(pick_numbers)
            
            # Calculate round probabilities (simplified)
            round_probabilities = {}
            if pick_numbers:
                for pick_num in pick_numbers:
                    round_num = ((pick_num - 1) // 12) + 1  # Assuming 12-team league #TODO: change to dynamically detect total num of teams
                    if round_num not in round_probabilities:
                        round_probabilities[round_num] = 0
                    round_probabilities[round_num] += 1 / len(pick_numbers)

            competing_teams = list(set(player_draft_teams[player_id]))

            availability = PlayerAvailability(player_id=player_id,
                                              availability_probability=availability_prob,
                                              average_pick_taken=avg_pick,
                                              earliest_pick=earliest_pick,
                                              latest_pick=latest_pick,
                                              pick_distribution=pick_distribution,
                                              round_probabilities=round_probabilities,
                                              urgency_score=1.0 - availability_prob,
                                              risk_assessment=min(1.0, (1.0 - availability_prob) * 1.5),  # Amplify risk
                                              simulated_by_teams=list(set(player_draft_teams[player_id])),
                                              competing_teams=competing_teams,
                                              simulation_id=f"sim_{int(time.time())}")
            
            availabilities[player_id] = availability

        return availabilities


    def _generate_draft_insights(self, 
                               availabilities: Dict[str, PlayerAvailability],
                               draft_state: DraftState,
                               parameters: SimulationParameters) -> Dict:
        """
        Generate high-level draft insights from simulation results.
        
        These insights help users understand market dynamics beyond
        individual player availability.
        """
        # Calculate position scarcity
        position_scarcity = {}
        position_urgency = defaultdict(list)
        
        for player_id, availability in availabilities.items():
            if player_id in self.players_db:
                position = self.players_db[player_id].position.value
                position_urgency[position].append(availability.urgency_score)
        
        for position, urgency_scores in position_urgency.items():
            if urgency_scores:
                position_scarcity[position] = np.mean(urgency_scores)
            else:
                position_scarcity[position] = 0.0
        
        # Identify likely position runs
        likely_runs = []
        for position, scarcity in position_scarcity.items():
            if scarcity > 0.6:  # High urgency = likely run
                likely_runs.append(position)
        
        # Identify value opportunities (players likely to fall)
        value_opportunities = []
        for player_id, availability in availabilities.items():
            if player_id in self.players_db:
                player = self.players_db[player_id]
                expected_pick = availability.average_pick_taken
                
                # If player is expected to go significantly later than ADP
                if expected_pick > player.adp + 10:
                    value_opportunities.append(player_id)
        
        return {
            'position_scarcity': position_scarcity,
            'likely_runs': likely_runs,
            'value_opportunities': value_opportunities[:10]  # Top 10
        }

    def _create_immediate_availability_result(self, 
                                            simulation_id: str,
                                            draft_state: DraftState,
                                            parameters: SimulationParameters,
                                            players: List[Player]) -> SimulationResult:
        """
        Create result when user is currently on the clock.
        
        No simulation needed - all available players are immediately available.
        """
        availabilities = {}
        
        for player in players:
            availability = PlayerAvailability(
                player_id=player.id,
                availability_probability=1.0,  # Available now
                average_pick_taken=draft_state.current_pick,
                earliest_pick=draft_state.current_pick,
                latest_pick=draft_state.current_pick,
                pick_distribution={draft_state.current_pick: 1.0},
                round_probabilities={
                    ((draft_state.current_pick - 1) // draft_state.team_count) + 1: 1.0
                },
                urgency_score=1.0,  # Maximum urgency - pick now!
                risk_assessment=0.0,  # No risk - available now
                simulated_by_teams=[],
                competing_teams=[],
                simulation_id=simulation_id
            )
            availabilities[player.id] = availability
        
        return SimulationResult(
            simulation_id=simulation_id,
            draft_id=draft_state.draft_id,
            parameters=parameters,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            execution_time_ms=1.0,  # Instant
            player_availabilities=availabilities,
            position_scarcity={},
            likely_runs=[],
            value_opportunities=[],
            converged=True,
            convergence_iteration=0,
            confidence_interval=1.0,
            current_pick=draft_state.current_pick,
            picks_simulated=0,
            user_team_id=draft_state.user_team_id
        )

    async def close(self):
        self.thread_pool.shutdown(wait=True)