"""
Snake draft order calculation utilities.

Handles the complex logic of determining pick orders in snake drafts,
which is essential for accurate simulation and user experience.
"""

from typing import Optional, List, Tuple
import math

class SnakeDraftCalculator:
    """
    Utility class for snake draft order calculations.
    
    Snake drafts reverse direction each round:
    Round 1: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    Round 2: 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    Round 3: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    
    This class handles all the math for determining who picks when.
    """
    
    def get_picking_team(self, 
                        pick_number: int,
                        team_count: int,
                        is_snake_draft: bool = True) -> str:
        """
        Determine which team picks at a given pick number.
        
        Args:
            pick_number: Overall pick number (1-based)
            team_count: Number of teams in draft
            is_snake_draft: Whether this is a snake draft
        
        Returns:
            Team ID (1-based string)
        """
        if pick_number < 1:
            raise ValueError("Pick number must be >= 1")
        
        if not is_snake_draft:
            # Linear draft - same order every round
            team_index = ((pick_number - 1) % team_count)
            return str(team_index + 1)
        
        # Snake draft logic
        round_number = ((pick_number - 1) // team_count) + 1
        pick_in_round = ((pick_number - 1) % team_count) + 1
        
        if round_number % 2 == 1:
            # Odd rounds: normal order (1, 2, 3, ...)
            team_index = pick_in_round - 1
        else:
            # Even rounds: reverse order (..., 3, 2, 1)
            team_index = team_count - pick_in_round
        
        return str(team_index + 1)
    
    def picks_until_team_turn(self, 
                             current_pick: int,
                             team_count: int,
                             target_team_index: int,
                             is_snake_draft: bool = True) -> int:
        """
        Calculate how many picks until a specific team's next turn.
        
        This is crucial for simulation scope - we only need to simulate
        until the user's next pick.
        
        Args:
            current_pick: Current pick number in draft
            team_count: Number of teams
            target_team_index: Target team index (0-based)
            is_snake_draft: Whether this is snake format
        
        Returns:
            Number of picks until target team's turn
        """
        if not is_snake_draft:
            # Linear draft - easy calculation
            current_team_index = ((current_pick - 1) % team_count)
            
            if current_team_index <= target_team_index:
                return target_team_index - current_team_index
            else:
                return team_count - current_team_index + target_team_index
        
        # Snake draft - more complex
        return self._calculate_snake_picks_until_turn(
            current_pick, team_count, target_team_index
        )
    
    def _calculate_snake_picks_until_turn(self, 
                                        current_pick: int,
                                        team_count: int,
                                        target_team_index: int) -> int:
        """
        Calculate picks until turn in snake draft.
        
        This handles the complexity of snake draft order reversals.
        """
        # Find target team's next pick
        next_pick = self._find_next_pick_for_team(
            current_pick, team_count, target_team_index
        )
        
        return max(0, next_pick - current_pick)
    
    def _find_next_pick_for_team(self, 
                               current_pick: int,
                               team_count: int,
                               target_team_index: int) -> int:
        """
        Find the next pick number for a specific team.
        
        Searches forward through pick numbers until finding the target team.
        """
        # Start checking from current pick
        check_pick = current_pick
        max_checks = team_count * 2  # Don't search more than 2 rounds
        
        for _ in range(max_checks):
            picking_team_id = self.get_picking_team(check_pick, team_count, True)
            picking_team_index = int(picking_team_id) - 1  # Convert to 0-based
            
            if picking_team_index == target_team_index:
                return check_pick
            
            check_pick += 1
        
        # Shouldn't reach here in normal circumstances
        return current_pick + team_count
    
    def get_draft_position_info(self, 
                              pick_number: int,
                              team_count: int) -> Tuple[int, int, int]:
        """
        Get detailed position information for a pick.
        
        Args:
            pick_number: Overall pick number
            team_count: Number of teams
        
        Returns:
            Tuple of (round_number, pick_in_round, team_index)
        """
        round_number = ((pick_number - 1) // team_count) + 1
        pick_in_round = ((pick_number - 1) % team_count) + 1
        
        # Calculate team index for snake draft
        if round_number % 2 == 1:
            team_index = pick_in_round - 1  # 0-based
        else:
            team_index = team_count - pick_in_round  # 0-based
        
        return round_number, pick_in_round, team_index
    
    def calculate_pick_value_curve(self, 
                                 team_count: int,
                                 total_rounds: int) -> List[Tuple[int, float]]:
        """
        Calculate the value curve for draft positions.
        
        Earlier picks are more valuable, but snake draft creates
        interesting dynamics where late first round + early second
        can be very valuable.
        
        Returns:
            List of (pick_number, relative_value) tuples
        """
        pick_values = []
        
        for pick_num in range(1, (team_count * total_rounds) + 1):
            round_num, pick_in_round, team_index = self.get_draft_position_info(
                pick_num, team_count
            )
            
            # Base value decreases with overall pick number
            base_value = 1.0 / pick_num
            
            # Bonus for getting consecutive picks in snake draft
            consecutive_bonus = self._calculate_consecutive_bonus(
                round_num, pick_in_round, team_count
            )
            
            total_value = base_value * (1 + consecutive_bonus)
            pick_values.append((pick_num, total_value))
        
        return pick_values
    
    def _calculate_consecutive_bonus(self, 
                                   round_num: int,
                                   pick_in_round: int,
                                   team_count: int) -> float:
        """
        Calculate bonus value for having consecutive picks.
        
        In snake drafts, teams at the ends get consecutive picks
        (e.g., picks 12 and 13 in a 12-team draft).
        """
        # Check if this is a "turn" position (last or first in round)
        if pick_in_round == 1 or pick_in_round == team_count:
            return 0.1  # 10% bonus for consecutive picks
        
        return 0.0


# Example usage and testing
def example_snake_draft_usage():
    """Example of how to use the SnakeDraftCalculator."""
    
    calculator = SnakeDraftCalculator()
    team_count = 12
    
    print("Snake Draft Order Example (12 teams):")
    print("=" * 50)
    
    # Show first 3 rounds
    for pick in range(1, 37):  # 3 rounds * 12 teams
        team_id = calculator.get_picking_team(pick, team_count, True)
        round_num, pick_in_round, team_index = calculator.get_draft_position_info(
            pick, team_count
        )
        
        print(f"Pick {pick:2d}: Team {team_id} (Round {round_num}, Pick {pick_in_round})")
        
        if pick % team_count == 0:
            print("-" * 30)
    
    # Calculate picks until team 5's turn from various positions
    print("\nPicks until Team 5's turn:")
    for current in [1, 5, 12, 13, 24, 25]:
        picks_until = calculator.picks_until_team_turn(
            current, team_count, 4, True  # Team 5 = index 4
        )
        print(f"From pick {current}: {picks_until} picks until Team 5")


if __name__ == "__main__":
    example_snake_draft_usage()