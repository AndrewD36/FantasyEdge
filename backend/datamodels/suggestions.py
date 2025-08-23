"""
Draft suggestion models and response structures.

These models represent the output of your AI system - the actual
recommendations provided to users. Clean design here shows
you understand user experience and API design.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from .player import Player, PlayerPosition

class SuggestionReason(str, Enum):
    """Reasons why a player might be suggested."""
    
    # Value-based reasons
    VALUE_PICK = "value_pick"              # Below ADP
    TIER_BREAK = "tier_break"              # Last player in tier
    BEST_AVAILABLE = "best_available"      # Highest projected points
    
    # Need-based reasons
    POSITIONAL_NEED = "positional_need"    # Team needs this position
    DEPTH_NEED = "depth_need"              # Need depth at position
    BYE_WEEK_FILL = "bye_week_fill"        # Covers bye week
    
    # Scarcity reasons
    POSITION_SCARCITY = "position_scarcity" # Position getting thin
    LIKELY_GONE = "likely_gone"             # Won't be available later
    RUN_RISK = "run_risk"                   # Position run starting
    
    # Strategic reasons
    UPSIDE_PLAY = "upside_play"            # High ceiling player
    SAFE_FLOOR = "safe_floor"              # Low risk option
    HANDCUFF = "handcuff"                  # Backup to owned player

    # NEW: Simulation-based reasons
    UNLIKELY_AVAILABLE = "unlikely_available"           # <30% availability
    LAST_IN_TIER = "last_in_tier"                       # Final elite player at position
    POSITION_RUN_STARTING = "position_run_starting"     # Run detected
    MARKET_INEFFICIENCY = "market_inefficiency"         # ADP vs simulation mismatch
    LOW_COMPETITION = "low_competition"                 # Few teams competing
    HIGH_VARIANCE = "high_variance"                     # Boom/bust candidate

class ConfidenceLevel(str, Enum):
    HIGH = "high"      # 80%+ confidence
    MEDIUM = "medium"  # 60-80% confidence
    LOW = "low"        # <60% confidence

class SuggestionMetadata(BaseModel):
    """
    Metadata about how a suggestion was generated.
    
    This is crucial for debugging, explaining decisions,
    and improving the algorithm over time.
    """
    
    # Algorithm information
    algorithm_version: str = Field(..., description="Version of suggestion algorithm")
    ml_model_version: Optional[str] = Field(None, description="ML model version if used")
    
    # Scoring breakdown
    base_score: float = Field(..., description="Base player value score")
    need_multiplier: float = Field(..., description="Positional need multiplier")
    scarcity_bonus: float = Field(..., description="Scarcity bonus points")
    timing_adjustment: float = Field(..., description="Draft timing adjustment")
    
    # Confidence and risk
    confidence_level: ConfidenceLevel = Field(..., description="Algorithm confidence")
    risk_assessment: float = Field(..., ge=0.0, le=1.0, description="Risk score (0=safe, 1=risky)")
    
    # Context
    generated_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    draft_context: Dict[str, Any] = Field(default_factory=dict, description="Draft state context")
    
    class Config:
        use_enum_values = True

class DraftSuggestion(BaseModel):
    # Player information
    player: Player = Field(..., description="The suggested player")
    
    # Suggestion scoring
    overall_score: float = Field(..., ge=0.0, description="Overall suggestion score")
    rank: int = Field(..., ge=1, description="Rank among current suggestions")
    
    # Reasoning
    primary_reason: SuggestionReason = Field(..., description="Main reason for suggestion")
    secondary_reasons: List[SuggestionReason] = Field(
        default_factory=list, 
        description="Additional supporting reasons")
    
    # Human-readable explanation
    explanation: str = Field(..., description="Natural language explanation")
    short_explanation: str = Field(..., description="Brief explanation for mobile/quick view")
    
    # NEW: Simulation-derived fields
    availability_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability player available at user's next turn")
    
    simulation_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Statistical confidence in availability prediction")
    
    competing_teams: List[str] = Field(
        default_factory=list,
        description="Teams likely to draft this player")
    
    expected_draft_round: Optional[int] = Field(
        None, description="Expected round when player will be drafted")
    
    # NEW: Market context
    position_run_risk: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Risk of position run affecting availability")
    
    value_trend: str = Field(
        "stable", description="Whether player is trending up/down/stable")

    projected_value: float = Field(..., description="Expected fantasy points")
    value_vs_adp: float = Field(..., description="Value compared to ADP (positive = good value)")
    
    # Risk assessment
    floor_projection: float = Field(..., description="Conservative projection")
    ceiling_projection: float = Field(..., description="Optimistic projection")
    bust_risk: float = Field(..., ge=0.0, le=1.0, description="Risk of significant underperformance")
    
    # Metadata
    metadata: SuggestionMetadata = Field(..., description="Algorithm metadata")
    
    class Config:
        use_enum_values = True

    @property
    def is_high_confidence(self) -> bool:
        return self.metadata.confidence_level == ConfidenceLevel.HIGH
    
    @property
    def is_value_pick(self) -> bool:
        return (self.primary_reason == SuggestionReason.VALUE_PICK or SuggestionReason.VALUE_PICK in self.secondary_reasons)
    
    @property
    def urgency_score(self) -> float:
        return 1.0 - self.availability_probability
    
class SuggestionRequest(BaseModel):
    draft_id: str = Field(..., description="Draft to get suggestions for")
    count: int = Field(10, ge=1, le=50, description="Number of suggestions to return")
    position_filter: Optional[PlayerPosition] = Field(None, description="Filter by position")
    min_confidence: Optional[ConfidenceLevel] = Field(None, description="Minimum confidence level")
    exclude_positions: List[PlayerPosition] = Field(
        default_factory=list, 
        description="Positions to exclude"
    )
    
    class Config:
        use_enum_values = True

class SuggestionResponse(BaseModel):
    # Core suggestions
    suggestions: List[DraftSuggestion] = Field(..., description="Ordered list of suggestions")
    
    # Request context
    requested_count: int = Field(..., description="Number of suggestions requested")
    actual_count: int = Field(..., description="Number of suggestions returned")
    
    # Draft context
    draft_id: str = Field(..., description="Draft ID these suggestions are for")
    current_pick: int = Field(..., description="Current pick number")
    user_on_clock: bool = Field(..., description="Whether user is currently picking")
    picks_until_user: int = Field(..., description="Picks until user's turn")
    
    # Market insights
    trending_up: List[str] = Field(
        default_factory=list, 
        description="Player IDs trending up in value"
    )
    trending_down: List[str] = Field(
        default_factory=list, 
        description="Player IDs trending down in value"
    )
    position_runs: Dict[str, int] = Field(
        default_factory=dict, 
        description="Recent picks by position (last 5 picks)"
    )
    
    # Performance metadata
    generation_time_ms: float = Field(..., description="Time to generate suggestions")
    cache_hit: bool = Field(False, description="Whether results came from cache")
    algorithm_version: str = Field(..., description="Algorithm version used")
    
    # Generated timestamp
    generated_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    
    class Config:
        use_enum_values = True

    @property
    def top_suggestion(self) -> Optional[DraftSuggestion]:
        return self.suggestions[0] if self.suggections else None
    
    @property
    def high_confidence_suggestions(self) -> List[DraftSuggestion]:
        return [suggestion for suggestion in self.suggestions if suggestion.is_high_confidence]
    
    @property
    def value_picks(self) -> List[DraftSuggestion]:
        return [suggestion for suggestion in self.suggestions if suggestion.is_value_pick]
    
class SuggestionFeedback(BaseModel):
    draft_id: str
    suggestion_id: str
    player_id: str
    feedback_type: str  # "helpful", "not_helpful", "wrong", "picked"
    user_comment: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now(timezone.utc))