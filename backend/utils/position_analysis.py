"""
Position analysis utilities for draft insights and scarcity calculations.

Provides tools for analyzing positional value, scarcity, and market dynamics
that inform both simulation behavior and user recommendations.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np

from ..models.player import Player, PlayerPosition
from ..models.draft_state import DraftState


class PositionAnalyzer:
    """
    Analyzes positional dynamics in fantasy drafts.
    
    Key functions:
    1. Calculate position scarcity scores
    2. Identify tier breaks and value cliffs
    3. Analyze draft flow and position runs
    4. Evaluate positional value curves
    """
    
    def __init__(self, players_db: Dict[str, Player]):
        """
        Initialize position analyzer with player database.
        
        Args:
            players_db: Complete player database
        """
        self.players_db = players_db
        self.position_tiers = self._calculate_position_tiers()
        self.value_curves = self._calculate_position_value_curves()
    
    def _calculate_position_tiers(self) -> Dict[PlayerPosition, List[List[str]]]:
        """
        Group players into tiers by position based on projected points.
        
        Tiers help identify value cliffs and scarcity points.
        """
        position_tiers = defaultdict(list)
        
        # Group players by position
        position_players = defaultdict(list)
        for player in self.players_db.values():
            position_players[player.position].append(player)
        
        # Create tiers for each position
        for position, players in position_players.items():
            # Sort by projected points
            players.sort(key=lambda p: p.projected_points, reverse=True)
            
            # Create tiers based on point drops
            tiers = []
            current_tier = []
            
            for i, player in enumerate(players):
                if i == 0:
                    current_tier.append(player.id)
                    continue
                
                # Check for significant point drop (tier break)
                point_drop = players[i-1].projected_points - player.projected_points
                avg_drop = self._calculate_average_drop(players[:i])
                
                if point_drop > avg_drop * 1.5:  # Significant tier break
                    tiers.append([p for p in current_tier])
                    current_tier = [player.id]
                else:
                    current_tier.append(player.id)
            
            # Add final tier
            if current_tier:
                tiers.append(current_tier)
            
            position_tiers[position] = tiers
        
        return position_tiers
    
    def _calculate_average_drop(self, players: List[Player]) -> float:
        """Calculate average point drop between consecutive players."""
        if len(players) < 2:
            return 0.0
        
        drops = []
        for i in range(1, len(players)):
            drop = players[i-1].projected_points - players[i].projected_points
            drops.append(drop)
        
        return np.mean(drops)
    
    def _calculate_position_value_curves(self) -> Dict[PlayerPosition, List[Tuple[int, float]]]:
        """
        Calculate value curves showing how player value decreases by position.
        
        Helps identify positions with steep vs gradual value drops.
        """
        value_curves = {}
        
        for position in PlayerPosition:
            position_players = [p for p in self.players_db.values() 
                              if p.position == position]
            position_players.sort(key=lambda p: p.projected_points, reverse=True)
            
            curve_points = []
            for i, player in enumerate(position_players[:50]):  # Top 50 at position
                rank = i + 1
                value = player.projected_points
                curve_points.append((rank, value))
            
            value_curves[position] = curve_points
        
        return value_curves
    
    def calculate_position_scarcity(self, 
                                  draft_state: DraftState,
                                  position: PlayerPosition) -> float:
        """
        Calculate current scarcity score for a position (0-1 scale).
        
        Higher scores indicate more urgent/scarce positions.
        
        Args:
            draft_state: Current draft state
            position: Position to analyze
        
        Returns:
            Scarcity score from 0.0 (abundant) to 1.0 (very scarce)
        """
        # Get remaining players at position
        remaining_players = [
            p for p in self.players_db.values()
            if p.position == position and p.id not in draft_state.drafted_players
        ]
        
        if not remaining_players:
            return 1.0  # Maximum scarcity - none left
        
        # Sort by quality (projected points)
        remaining_players.sort(key=lambda p: p.projected_points, reverse=True)
        
        # Calculate multiple scarcity factors
        factors = []
        
        # 1. Absolute quantity factor
        total_at_position = len([p for p in self.players_db.values() if p.position == position])
        remaining_count = len(remaining_players)
        quantity_factor = 1.0 - (remaining_count / total_at_position)
        factors.append(quantity_factor)
        
        # 2. Quality factor (how many elite players left?)
        elite_remaining = len([p for p in remaining_players if p.tier <= 2])
        total_elite = len([p for p in self.players_db.values() 
                          if p.position == position and p.tier <= 2])
        if total_elite > 0:
            quality_factor = 1.0 - (elite_remaining / total_elite)
            factors.append(quality_factor)
        
        # 3. Tier break proximity factor
        tier_break_factor = self._calculate_tier_break_proximity(
            remaining_players, draft_state.current_pick
        )
        factors.append(tier_break_factor)
        
        # 4. Position-specific scarcity multipliers
        position_multipliers = {
            PlayerPosition.QB: 0.7,   # Less scarce overall
            PlayerPosition.RB: 1.2,   # More scarce
            PlayerPosition.WR: 0.9,   # Moderate scarcity
            PlayerPosition.TE: 1.1,   # Quite scarce at top
            PlayerPosition.K: 0.3,    # Not really scarce
            PlayerPosition.DST: 0.5   # Moderate scarcity
        }
        
        base_scarcity = np.mean(factors)
        multiplier = position_multipliers.get(position, 1.0)
        
        return min(1.0, base_scarcity * multiplier)
    
    def _calculate_tier_break_proximity(self, 
                                      remaining_players: List[Player],
                                      current_pick: int) -> float:
        """
        Calculate how close we are to a significant tier break.
        
        Returns higher values when approaching a cliff in player quality.
        """
        if len(remaining_players) < 2:
            return 1.0
        
        # Look at top remaining players and their point drops
        top_players = remaining_players[:10]  # Top 10 remaining
        
        max_drop = 0.0
        for i in range(1, len(top_players)):
            drop = top_players[i-1].projected_points - top_players[i].projected_points
            max_drop = max(max_drop, drop)
        
        # Normalize drop (15+ points = significant tier break)
        return min(1.0, max_drop / 15.0)
    
    def analyze_draft_flow(self, draft_state: DraftState) -> Dict[str, any]:
        """
        Analyze recent draft flow for position runs and patterns.
        
        Args:
            draft_state: Current draft state
        
        Returns:
            Dictionary with flow analysis insights
        """
        if not draft_state.completed_picks:
            return {'recent_positions': [], 'position_runs': {}, 'momentum': {}}
        
        # Get recent picks (last 10)
        recent_picks = draft_state.completed_picks[-10:]
        
        # Count positions in recent picks
        recent_positions = []
        for pick in recent_picks:
            if pick.player_id and pick.player_id in self.players_db:
                position = self.players_db[pick.player_id].position.value
                recent_positions.append(position)
        
        # Detect position runs
        position_runs = self._detect_position_runs(recent_positions)
        
        # Calculate momentum for each position
        momentum = self._calculate_position_momentum(recent_positions)
        
        return {
            'recent_positions': recent_positions,
            'position_runs': position_runs,
            'momentum': momentum,
            'picks_analyzed': len(recent_picks)
        }
    
    def _detect_position_runs(self, recent_positions: List[str]) -> Dict[str, Dict]:
        """
        Detect position runs in recent picks.
        
        A run is defined as 3+ picks of the same position in a 5-pick window.
        """
        runs = {}
        window_size = 5
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            max_in_window = 0
            
            # Check each window
            for i in range(len(recent_positions) - window_size + 1):
                window = recent_positions[i:i + window_size]
                count = window.count(position)
                max_in_window = max(max_in_window, count)
            
            if max_in_window >= 3:
                runs[position] = {
                    'intensity': max_in_window,
                    'is_active': recent_positions[-3:].count(position) >= 2,
                    'probability_continues': min(1.0, max_in_window / 5.0)
                }
        
        return runs
    
    def _calculate_position_momentum(self, recent_positions: List[str]) -> Dict[str, float]:
        """
        Calculate drafting momentum for each position.
        
        Momentum indicates whether a position is being drafted more/less recently.
        """
        if len(recent_positions) < 4:
            return {}
        
        momentum = {}
        
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            # Compare recent half vs earlier half
            mid_point = len(recent_positions) // 2
            earlier_half = recent_positions[:mid_point]
            recent_half = recent_positions[mid_point:]
            
            earlier_count = earlier_half.count(position)
            recent_count = recent_half.count(position)
            
            # Calculate momentum score (-1 to 1)
            total_picks = len(recent_positions)
            earlier_rate = earlier_count / mid_point if mid_point > 0 else 0
            recent_rate = recent_count / (total_picks - mid_point) if (total_picks - mid_point) > 0 else 0
            
            if earlier_rate + recent_rate > 0:
                momentum_score = (recent_rate - earlier_rate) / (earlier_rate + recent_rate + 0.1)
            else:
                momentum_score = 0.0
            
            momentum[position] = momentum_score
        
        return momentum
    
    def get_position_recommendations(self, 
                                   draft_state: DraftState,
                                   user_team_roster: List[str]) -> Dict[str, Dict]:
        """
        Generate position-specific recommendations for the user.
        
        Args:
            draft_state: Current draft state
            user_team_roster: User's current roster (player IDs)
        
        Returns:
            Dictionary with recommendations for each position
        """
        recommendations = {}
        
        # Analyze user's current positional needs
        user_position_counts = self._count_user_positions(user_team_roster)
        
        for position in PlayerPosition:
            pos_name = position.value
            
            # Calculate need score
            need_score = self._calculate_user_position_need(
                position, user_position_counts, draft_state
            )
            
            # Calculate scarcity score
            scarcity_score = self.calculate_position_scarcity(draft_state, position)
            
            # Calculate value opportunity score
            value_score = self._calculate_position_value_opportunity(
                position, draft_state
            )
            
            # Generate recommendation
            recommendations[pos_name] = {
                'need_score': need_score,
                'scarcity_score': scarcity_score,
                'value_score': value_score,
                'urgency': (need_score + scarcity_score) / 2,
                'recommendation': self._generate_position_recommendation(
                    need_score, scarcity_score, value_score
                )
            }
        
        return recommendations
    
    def _count_user_positions(self, roster: List[str]) -> Dict[str, int]:
        """Count how many players the user has at each position."""
        counts = defaultdict(int)
        
        for player_id in roster:
            if player_id in self.players_db:
                position = self.players_db[player_id].position.value
                counts[position] += 1
        
        return counts
    
    def _calculate_user_position_need(self, 
                                    position: PlayerPosition,
                                    current_counts: Dict[str, int],
                                    draft_state: DraftState) -> float:
        """Calculate how much the user needs this position."""
        pos_name = position.value
        current_count = current_counts.get(pos_name, 0)
        
        # Standard roster requirements
        requirements = {
            'QB': 1,
            'RB': 2,
            'WR': 3,
            'TE': 1,
            'K': 1,
            'DST': 1
        }
        
        required = requirements.get(pos_name, 0)
        
        if current_count < required:
            return 1.0  # High need
        elif current_count < required + 1 and pos_name in ['RB', 'WR']:
            return 0.6  # Moderate need for depth
        else:
            return 0.2  # Low need
    
    def _calculate_position_value_opportunity(self, 
                                            position: PlayerPosition,
                                            draft_state: DraftState) -> float:
        """Calculate value opportunity score for position."""
        # Get remaining players at position
        remaining = [
            p for p in self.players_db.values()
            if p.position == position and p.id not in draft_state.drafted_players
        ]
        
        if not remaining:
            return 0.0
        
        # Look for players significantly below their ADP
        current_pick = draft_state.current_pick
        value_opportunities = 0
        
        for player in remaining:
            if player.adp < current_pick - 5:  # Available 5+ picks after ADP
                value_opportunities += 1
        
        # Normalize to 0-1 scale
        return min(1.0, value_opportunities / 5.0)
    
    def _generate_position_recommendation(self, 
                                        need: float,
                                        scarcity: float,
                                        value: float) -> str:
        """Generate human-readable recommendation."""
        if need > 0.8 and scarcity > 0.7:
            return "URGENT - High need and very scarce"
        elif need > 0.8:
            return "HIGH PRIORITY - Position of need"
        elif scarcity > 0.8:
            return "CONSIDER - Position getting very thin"
        elif value > 0.7:
            return "VALUE OPPORTUNITY - Good players falling"
        elif need > 0.5 or scarcity > 0.5:
            return "MODERATE PRIORITY - Keep on radar"
        else:
            return "LOW PRIORITY - Can wait"


# Example usage
def example_position_analysis():
    """Example of how to use position analysis tools."""
    
    # This would use real player data in practice
    sample_players = {}  # Would load from database
    
    analyzer = PositionAnalyzer(sample_players)
    
    # Example draft state
    sample_draft_state = None  # Would be real draft state
    
    if sample_draft_state:
        # Analyze position scarcity
        rb_scarcity = analyzer.calculate_position_scarcity(
            sample_draft_state, PlayerPosition.RB
        )
        print(f"RB Scarcity Score: {rb_scarcity:.2f}")
        
        # Analyze draft flow
        flow_analysis = analyzer.analyze_draft_flow(sample_draft_state)
        print(f"Recent positions: {flow_analysis['recent_positions']}")
        print(f"Position runs: {flow_analysis['position_runs']}")


if __name__ == "__main__":
    example_position_analysis()