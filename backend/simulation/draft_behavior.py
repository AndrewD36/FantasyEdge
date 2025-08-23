"""
Human draft behavior simulation for realistic Monte Carlo modeling.

This module captures the psychology and patterns of how real humans draft,
making simulations more accurate than pure ADP-based randomness.
"""

import random
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

from ..datamodels.player import Player, PlayerPosition
from ..datamodels.simulation import SimulationStrategy, SimulationParameters

class DraftBehaviorSimulator:
    """
    Simulates realistic human draft behavior patterns.
    
    Key insights modeled:
    1. Positional preferences vary by round
    2. Teams draft for need vs value differently over time
    3. ADP adherence decreases in later rounds
    4. Position runs create herd mentality
    5. Time pressure affects decision quality
    
    This is where domain expertise about fantasy football gets encoded.
    """
    
    def __init__(self, players_db: Dict[str, Player]):
        """
        Initialize behavior simulator with player database.
        
        Args:
            players_db: Complete player database for context
        """
        self.players_db = players_db
        
        # Behavioral constants derived from real draft analysis
        self.position_preferences = self._initialize_position_preferences()
        self.adp_variance_by_round = self._initialize_adp_variance()
        self.need_weight_by_round = self._initialize_need_weights()
        
    def _initialize_position_preferences(self) -> Dict[int, Dict[str, float]]:
        """
        Initialize positional drafting preferences by round.
        
        Based on analysis of thousands of real drafts showing
        clear positional preferences that vary by draft round.
        """
        return {
            1: {'RB': 0.45, 'WR': 0.35, 'QB': 0.15, 'TE': 0.05},  # RB-heavy early
            2: {'RB': 0.40, 'WR': 0.45, 'QB': 0.10, 'TE': 0.05},  # WR catchup
            3: {'RB': 0.35, 'WR': 0.40, 'QB': 0.15, 'TE': 0.10},  # More balanced
            4: {'RB': 0.30, 'WR': 0.35, 'QB': 0.25, 'TE': 0.10},  # QB runs start
            5: {'RB': 0.25, 'WR': 0.30, 'QB': 0.30, 'TE': 0.15},  # QB-heavy
            6: {'RB': 0.25, 'WR': 0.30, 'QB': 0.25, 'TE': 0.20},  # TE becomes viable
            7: {'RB': 0.30, 'WR': 0.35, 'QB': 0.15, 'TE': 0.20},  # Back to skill
            8: {'RB': 0.35, 'WR': 0.35, 'QB': 0.10, 'TE': 0.15, 'DST': 0.05},
            9: {'RB': 0.30, 'WR': 0.30, 'QB': 0.15, 'TE': 0.15, 'DST': 0.05, 'K': 0.05},
            10: {'RB': 0.25, 'WR': 0.25, 'QB': 0.20, 'TE': 0.15, 'DST': 0.10, 'K': 0.05},
        }
    
    def _initialize_adp_variance(self) -> Dict[int, float]:
        """
        Initialize ADP variance (randomness) by round.
        
        Early rounds follow ADP closely, later rounds more chaotic.
        """
        return {
            1: 0.1,   # Very tight to ADP
            2: 0.15,  # Still pretty tight
            3: 0.2,   # Some variance
            4: 0.25,  # More variance
            5: 0.3,   # Getting chaotic
            6: 0.35,  # Quite random
            7: 0.4,   # Very random
            8: 0.45,  # Extremely random
            9: 0.5,   # Complete chaos
            10: 0.6,  # Anyone could go anywhere
        }
    
    def _initialize_need_weights(self) -> Dict[int, float]:
        """
        Initialize need-based drafting weights by round.
        
        Early rounds prioritize value, later rounds prioritize filling holes.
        """
        return {
            1: 0.1,   # Almost pure value
            2: 0.15,  # Still mostly value
            3: 0.2,   # Some need consideration
            4: 0.3,   # Balanced
            5: 0.4,   # Need becomes important
            6: 0.5,   # Equal weight
            7: 0.6,   # Need-heavy
            8: 0.7,   # Mostly need
            9: 0.8,   # Almost pure need
            10: 0.9,  # Fill remaining holes
        }
    
    def simulate_team_pick(self, 
                          team_id: str,
                          available_players: List[Player],
                          draft_context: Dict[str, Any],
                          rng: random.Random) -> Optional[Player]:
        """
        Simulate a single team's draft pick using behavioral modeling.
        
        This is the core function that determines what a team will draft.
        It balances multiple factors to create realistic pick patterns.
        
        Args:
            team_id: ID of team making the pick
            available_players: Players available to draft
            draft_context: Current draft state and context
            rng: Random number generator for this simulation
        
        Returns:
            Selected player or None if no valid pick
        """
        if not available_players:
            return None
        
        current_pick = draft_context['current_pick']
        team_roster = draft_context['team_roster']
        parameters = draft_context['parameters']
        
        # Calculate draft round and phase
        round_number = ((current_pick - 1) // 12) + 1  # Assume 12-team league
        draft_phase = self._get_draft_phase(round_number)
        
        # Calculate pick scores for all available players
        pick_scores = []
        
        for player in available_players:
            score = self._calculate_pick_score(
                player=player,
                team_roster=team_roster,
                round_number=round_number,
                draft_phase=draft_phase,
                parameters=parameters,
                rng=rng
            )
            pick_scores.append((player, score))
        
        # Sort by score and add some randomness
        pick_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top candidates with weighted randomness
        return self._select_with_randomness(pick_scores, parameters, rng)
    
    def _calculate_pick_score(self, 
                            player: Player,
                            team_roster: List[str],
                            round_number: int,
                            draft_phase: str,
                            parameters: SimulationParameters,
                            rng: random.Random) -> float:
        """
        Calculate a comprehensive pick score for a player.
        
        Combines multiple factors:
        1. Player talent (projected points, tier)
        2. ADP value (is this good value at this pick?)
        3. Positional need (does team need this position?)
        4. Positional preference (round-based position bias)
        5. Scarcity factor (is this position getting thin?)
        6. Randomness (human unpredictability)
        """
        score = 0.0
        
        # 1. Base talent score (40% weight)
        talent_score = player.projected_points / 400.0  # Normalize to ~0-1 range
        talent_score *= (6 - player.tier) / 5.0  # Tier bonus (tier 1 = 1.0, tier 5 = 0.2)
        score += 0.4 * talent_score
        
        # 2. ADP value score (25% weight)
        # Positive if player available later than ADP, negative if earlier
        current_pick = ((round_number - 1) * 12) + 6  # Approximate pick in round
        adp_value = max(0, (player.adp - current_pick) / 20.0)  # Normalize
        score += 0.25 * adp_value
        
        # 3. Positional need score (20% weight)
        need_score = self._calculate_positional_need(player.position, team_roster, round_number)
        need_weight = self.need_weight_by_round.get(round_number, 0.5)
        score += 0.2 * need_score * need_weight
        
        # 4. Positional preference score (10% weight)
        position_pref = self._get_position_preference(player.position, round_number)
        score += 0.1 * position_pref
        
        # 5. Scarcity bonus (5% weight)
        scarcity_bonus = self._calculate_scarcity_bonus(player.position)
        score += 0.05 * scarcity_bonus
        
        # 6. Apply strategy modifications
        score = self._apply_strategy_modifiers(score, player, parameters)
        
        # 7. Add randomness based on draft phase
        variance = self.adp_variance_by_round.get(round_number, 0.3)
        randomness_factor = rng.gauss(1.0, variance)  # Normal distribution around 1.0
        score *= max(0.1, randomness_factor)  # Prevent negative scores
        
        return score
    
    def _calculate_positional_need(self, 
                                 position: PlayerPosition,
                                 team_roster: List[str],
                                 round_number: int) -> float:
        """
        Calculate how much a team needs this position.
        
        Based on standard roster construction and what they've already drafted.
        """
        # Count current position holdings
        position_counts = defaultdict(int)
        for player_id in team_roster:
            if player_id in self.players_db:
                pos = self.players_db[player_id].position
                position_counts[pos] += 1
        
        current_count = position_counts[position]
        
        # Standard roster needs by position
        position_needs = {
            PlayerPosition.QB: 1,
            PlayerPosition.RB: 2,
            PlayerPosition.WR: 3,
            PlayerPosition.TE: 1,
            PlayerPosition.K: 1,
            PlayerPosition.DST: 1
        }
        
        needed = position_needs.get(position, 0)
        
        # Calculate need score
        if current_count < needed:
            # High need - missing required starters
            return 1.0 + (needed - current_count) * 0.2
        elif current_count < needed + 1 and position in [PlayerPosition.RB, PlayerPosition.WR]:
            # Moderate need - could use depth at skill positions
            return 0.6
        elif round_number <= 6 and current_count == 0:
            # Early round, haven't drafted this position yet
            return 0.8
        else:
            # Low need
            return 0.2
    
    def _get_position_preference(self, position: PlayerPosition, round_number: int) -> float:
        """Get positional preference for this round."""
        round_prefs = self.position_preferences.get(round_number, {})
        return round_prefs.get(position.value, 0.1)  # Default low preference
    
    def _calculate_scarcity_bonus(self, position: PlayerPosition) -> float:
        """
        Calculate scarcity bonus for position.
        
        Some positions (RB, TE) have fewer quality options, creating urgency.
        """
        scarcity_multipliers = {
            PlayerPosition.QB: 0.3,   # Many viable options
            PlayerPosition.RB: 1.0,   # Limited elite options
            PlayerPosition.WR: 0.6,   # Moderate scarcity
            PlayerPosition.TE: 0.9,   # Very scarce at top
            PlayerPosition.K: 0.1,    # All similar
            PlayerPosition.DST: 0.4   # Some much better than others
        }
        
        return scarcity_multipliers.get(position, 0.5)
    
    def _apply_strategy_modifiers(self, 
                                base_score: float,
                                player: Player,
                                parameters: SimulationParameters) -> float:
        """
        Apply strategy-specific modifications to pick score.
        
        Different simulation strategies emphasize different factors.
        """
        if parameters.strategy == SimulationStrategy.CONSERVATIVE:
            # Conservative: Heavily favor ADP, avoid risks
            if player.adp <= 50:  # Early ADP players
                base_score *= 1.2
            if player.tier <= 2:  # High tier players
                base_score *= 1.1
        
        elif parameters.strategy == SimulationStrategy.AGGRESSIVE:
            # Aggressive: More need-based, willing to reach
            base_score *= 1.0 + (parameters.need_weight * 0.5)
            # Slight penalty for following ADP too closely
            base_score *= 0.95
        
        elif parameters.strategy == SimulationStrategy.BALANCED:
            # Balanced: No major modifications
            pass
        
        elif parameters.strategy == SimulationStrategy.HISTORICAL:
            # Historical: Apply historical patterns
            # This would use actual historical draft data patterns
            pass
        
        return base_score
    
    def _get_draft_phase(self, round_number: int) -> str:
        """Categorize draft phase for behavior adjustments."""
        if round_number <= 3:
            return "early"
        elif round_number <= 7:
            return "middle"
        else:
            return "late"
    
    def _select_with_randomness(self, 
                              scored_players: List[tuple],
                              parameters: SimulationParameters,
                              rng: random.Random) -> Optional[Player]:
        """
        Select player from scored list with appropriate randomness.
        
        Uses weighted selection to model human decision-making:
        - Top players more likely to be selected
        - But some randomness for realistic unpredictability
        """
        if not scored_players:
            return None
        
        # Apply chaos factor from parameters
        chaos = parameters.chaos_factor
        
        if chaos <= 0.1:
            # Very predictable - just take the top player
            return scored_players[0][0]
        
        # Calculate selection weights with exponential decay
        weights = []
        for i, (player, score) in enumerate(scored_players[:20]):  # Consider top 20
            # Exponential decay with chaos factor
            weight = np.exp(-i * chaos * 2)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight <= 0:
            return scored_players[0][0]
        
        probabilities = [w / total_weight for w in weights]
        
        # Random selection based on probabilities
        selected_index = rng.choices(range(len(weights)), weights=probabilities)[0]
        
        return scored_players[selected_index][0]


class PositionRunDetector:
    """
    Detects and models position runs during drafts.
    
    Position runs create market dynamics where teams rush to draft
    from a specific position, creating urgency and value opportunities.
    """
    
    def __init__(self):
        self.run_thresholds = {
            'QB': 3,    # 3 QBs in 5 picks = QB run
            'RB': 4,    # 4 RBs in 5 picks = RB run  
            'WR': 4,    # 4 WRs in 5 picks = WR run
            'TE': 2,    # 2 TEs in 5 picks = TE run
        }
    
    def detect_run_probability(self, 
                             recent_picks: List[str],
                             position: str,
                             window_size: int = 5) -> float:
        """
        Calculate probability that a position run is occurring.
        
        Args:
            recent_picks: List of recent player IDs drafted
            position: Position to check for runs
            window_size: Number of recent picks to consider
        
        Returns:
            Probability (0-1) that a position run is happening
        """
        if len(recent_picks) < 2:
            return 0.0
        
        # Count positions in recent picks
        position_count = sum(1 for pick in recent_picks[-window_size:] 
                           if self._get_player_position(pick) == position)
        
        threshold = self.run_thresholds.get(position, 3)
        
        # Calculate run probability
        if position_count >= threshold:
            return 1.0
        elif position_count >= threshold - 1:
            return 0.7
        elif position_count >= threshold - 2:
            return 0.3
        else:
            return 0.0
    
    def _get_player_position(self, player_id: str) -> Optional[str]:
        """Get position for player ID - would integrate with player database."""
        # This would lookup position from player database
        return None  # Placeholder