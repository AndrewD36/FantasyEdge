"""
Monte Carlo simulation models and result structures.

These models capture the output of simulation runs and provide
structured data for decision making and caching.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from enum import Enum
from pydantic import BaseModel, Field

from .player import Player

class SimulationStrategy(str, Enum):
    CONSERVATIVE = "conservative"  # More predictable, ADP-focused
    AGGRESSIVE = "aggressive"     # More chaotic, need-based
    BALANCED = "balanced"         # Mix of both approaches
    HISTORICAL = "historical"     # Based on historical draft patterns

@dataclass(frozen=True)
class SimulationParameters:
    # Core simulation settings
    num_simulations: int = 1000
    strategy: SimulationStrategy = SimulationStrategy.BALANCED
    
    # Behavior modeling parameters
    adp_adherence: float = 0.7      # How closely teams follow ADP (0-1)
    need_weight: float = 0.3        # How much teams draft for need vs value
    chaos_factor: float = 0.1       # Random variation in picks
    
    # Draft context
    draft_phase: str = "early"      # early, middle, late
    time_pressure: bool = False     # Simulate time pressure effects
    
    # Advanced options
    position_run_probability: float = 0.15  # Chance of position runs
    sleeper_pick_chance: float = 0.05      # Chance of surprising picks

    def __post__init__(self):
        assert 0 <= self.adp_adherence <= 1, "ADP adherence must be 0-1"
        assert 0 <= self.need_weight <= 1, "Need weight must be 0-1"
        assert 0 <= self.chaos_factor <= 1, "Chaos factor must be 0-1"

    @property
    def cache_key(self) -> str:
        """Generate cache key for these parameters."""
        return f"sim_{self.num_simulations}_{self.strategy.value}_{self.adp_adherence}_{self.need_weight}"
    
class PlayerAvailability(BaseModel):
    player_id: str = Field(..., description="Player identifier")
    
    # Core availability metrics
    availability_probability: float = Field(..., ge=0.0, le=1.0, 
                                           description="Probability available at user's next pick")
    average_pick_taken: float = Field(..., description="Average pick number when drafted")
    earliest_pick: int = Field(..., description="Earliest pick taken in simulations")
    latest_pick: int = Field(..., description="Latest pick taken in simulations")
    
    # Distribution data
    pick_distribution: Dict[int, float] = Field(..., description="Pick number -> probability")
    round_probabilities: Dict[int, float] = Field(..., description="Round -> probability drafted")
    
    # Urgency metrics
    urgency_score: float = Field(..., ge=0.0, le=1.0, 
                                description="How urgent this pick is (1-availability)")
    risk_assessment: float = Field(..., ge=0.0, le=1.0,
                                  description="Risk of missing out on player")
    
    # Context data
    simulated_by_teams: List[str] = Field(..., description="Teams that drafted this player")
    competing_teams: List[str] = Field(..., description="Teams likely to draft this player")
    
    # Metadata
    simulation_id: str = Field(..., description="Simulation run identifier")
    generated_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    @property
    def is_likely_available(self) -> bool:
        return self.availability_probability > 0.6
    
    @property
    def is_urgent_pick(self) -> bool:
        return self.availability_probability > 0.3
    
    @property
    def expected_wait_rounds(self) -> float:
        total_weighted_rounds = sum(round_num * probability for round_num, probability in self.round_probabilities.items())
        return total_weighted_rounds
    
class SimulationResult(BaseModel):
    # Simulation metadata
    simulation_id: str = Field(..., description="Unique simulation identifier")
    draft_id: str = Field(..., description="Draft this simulation was run for")
    parameters: SimulationParameters = Field(..., description="Simulation configuration")
    
    # Timing data
    started_at: datetime = Field(..., description="When simulation started")
    completed_at: datetime = Field(..., description="When simulation completed")
    execution_time_ms: float = Field(..., description="Simulation runtime in milliseconds")
    
    # Core results
    player_availabilities: Dict[str, PlayerAvailability] = Field(..., 
                                                                 description="Player ID -> availability data")
    
    # Draft insights
    position_scarcity: Dict[str, float] = Field(..., description="Position -> scarcity score")
    likely_runs: List[str] = Field(..., description="Positions likely to have runs")
    value_opportunities: List[str] = Field(..., description="Players likely to fall in value")
    
    # Convergence data
    converged: bool = Field(..., description="Whether simulation converged")
    convergence_iteration: Optional[int] = Field(None, description="Iteration where convergence occurred")
    confidence_interval: float = Field(..., description="Statistical confidence in results")
    
    # Draft state context
    current_pick: int = Field(..., description="Pick number when simulation was run")
    picks_simulated: int = Field(..., description="Number of picks simulated")
    user_team_id: Optional[str] = Field(None, description="User's team in the draft")

    def get_player_availability(self, player_id: str) -> Optional[PlayerAvailability]:
        return self.player_availabilities.get(player_id)
    
    def get_urgent_picks(self, threshold: float = 0.3) -> List[PlayerAvailability]:
        return [availability for availability in self.player_availabilities.values() if availability.availability_probability < threshold]
    
    def get_safe_picks(self, threshold: float = 0.7) ->List[PlayerAvailability]:
        return [availability for availability in self.player_availabilities.values() if availability.availability_probability < threshold]
    
    def get_position_availability(self, position: str) -> List[PlayerAvailability]:
        # This would require player position lookup - simplified for now
        return list(self.player_availabilities.values())
    
    @property
    def total_players_analyzed(self) -> int:
        return len(self.player_availabilities)
    
    @property
    def average_confidence(self) -> float:
        if not self.player_availabilities:
            return 0.0
        
        confidences = []
        for availability in self.player_availabilities.values():
            # Calculate confidence based on probability distribution
            # More extreme probabilities (close to 0 or 1) = higher confidence
            prob = availability.availability_probability
            confidence = abs(prob - 0.5) * 2  # Convert to 0-1 scale
            confidences.append(confidence)

        return sum(confidences) / len(confidences)
    
class SimulationRequest(BaseModel):
    draft_id: str = Field(..., description="Draft to simulate")
    parameters: Optional[SimulationParameters] = Field(None, description="Simulation parameters")
    target_players: Optional[List[str]] = Field(None, description="Specific players to analyze")
    use_cache: bool = Field(True, description="Whether to use cached results")

    class config:
        schema_extra = {
            "example": {
                "draft_id": "draft_123456",
                "parameters": {
                    "num_simulations": 1000,
                    "strategy": "balanced",
                    "adp_adherence": 0.7
                },
                "target_players": ["player_1", "player_2"],
                "use_cache": True
            }
        }

class SimulationResponse(BaseModel):
    success: bool = Field(..., description="Whether simulation completed successfully")
    result: Optional[SimulationResult] = Field(None, description="Simulation results")
    cached: bool = Field(False, description="Whether results came from cache")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Performance metadata
    api_response_time_ms: float = Field(..., description="Total API response time")
    cache_hit: bool = Field(False, description="Whether cache was used")