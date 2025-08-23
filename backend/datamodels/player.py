"""
Player data model and related enums.

This is the core entity representing NFL players in our system.
Designed to be immutable and validation-focused for data integrity.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class PlayerPosition(str, Enum):
    QB ="QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    DST = "DST"
    K = "K"

class PlayerTier(int, Enum):
    ELITE = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    DEPTH = 5

@dataclass(frozen=True)
class Player:
    """
    Represents an NFL player with fantasy football relevant data.
    
    This is designed to be the single source of truth for player data.
    Immutable design prevents accidental modifications and enables caching.
    """

    #TODO: add Field properties to each variable below
    id: str
    name: str
    position: PlayerPosition

    team: str

    adp: float
    projected_points: float
    std_dev: float
    tier: PlayerTier
    bye_week: int

    sleeper_id: Optional[str]
    epsn_id: Optional[str]
    yahoo_id: Optional[str]

    vorp: Optional[float]

    @field_validator('team')
    def validate_team_abbreviation(cls, v):
        return v.upper()
    
    @field_validator('adp')
    def validate_adp(cls, v, values):
        postion = values.get('postion')
        if postion == PlayerPosition.DST and v < 150:
            raise ValueError("DST picks should have an ADP value greater than 150")
        if postion == PlayerPosition.K and v < 120:
            raise ValueError("Kickers should have an ADP value greater than 120")
        
        return v
    
    @field_validator('projected_points')
    def validate_realistic_projections(cls, v, values):
        """Enhanced validation for realistic fantasy projections."""
        position = values.get('position')
        
        # Position-specific validation ranges
        projection_ranges = {
            PlayerPosition.QB: (150, 450),   # QB scoring range
            PlayerPosition.RB: (50, 400),    # RB scoring range  
            PlayerPosition.WR: (40, 350),    # WR scoring range
            PlayerPosition.TE: (30, 250),    # TE scoring range
            PlayerPosition.K: (80, 150),     # Kicker range
            PlayerPosition.DST: (60, 200)    # Defense range
        }
        
        if position and position in projection_ranges:
            min_proj, max_proj = projection_ranges[position]
            if not (min_proj <= v <= max_proj):
                raise ValueError(f"{position.value} projection {v} outside realistic range {min_proj}-{max_proj}")
        
        return v
    
    def __str__(self) -> str:
        return f'{self.name} ({self.position.value}, {self.team})'
    
    def __repr__(self) -> str:
        return f'Player (id={self.id}, name={self.name}, position={self.position.value})'
    
    @property
    def is_skill_position(self) -> bool:
        return self.position in [PlayerPosition.QB, PlayerPosition.RB, PlayerPosition.WR, PlayerPosition.TE]
    
    @property
    def position_scarcity_tier(self) -> str:
        scarcity_tiers = {PlayerPosition.QB: 'low',
                          PlayerPosition.RB: 'high',
                          PlayerPosition.WR: 'medium',
                          PlayerPosition.TE: 'high',
                          PlayerPosition.DST: 'medium',
                          PlayerPosition.K: 'low'}
        return scarcity_tiers[self.position]
    
    @property
    def simulation_weight(self) -> float:
        """
        Weight for simulation selection probability.
        
        Combines ADP ranking with projected value for realistic draft modeling.
        Lower ADP + higher projections = higher weight in simulations.
        """
        # Inverse ADP (lower ADP = higher weight)
        adp_weight = 1.0 / max(1.0, self.adp)
        
        # Normalized projection weight
        projection_weight = self.projected_points / 400.0  # Normalize to ~0-1
        
        # Combine with slight ADP bias (70% ADP, 30% projections)
        return (0.7 * adp_weight) + (0.3 * projection_weight)

    @property 
    def draft_urgency_factor(self) -> float:
        """
        Calculate urgency factor for draft timing.
        
        Elite players (tier 1-2) have higher urgency as they disappear quickly.
        Later tier players have lower urgency as substitutes exist.
        """
        tier_urgency = {
            PlayerTier.ELITE: 1.0,    # Maximum urgency
            PlayerTier.HIGH: 0.8,     # High urgency  
            PlayerTier.MEDIUM: 0.6,   # Moderate urgency
            PlayerTier.LOW: 0.4,      # Low urgency
            PlayerTier.DEPTH: 0.2     # Minimal urgency
        }
        return tier_urgency.get(self.tier, 0.5)
    
class PlayerAPI(BaseModel):
    id: str
    name: str
    position: PlayerPosition
    team: str
    adp: float
    projected_points: float
    std_dev: float
    tier: PlayerTier
    bye_week: int
    sleeper_id: Optional[str] = None
    epsn_id: Optional[str] = None
    yahoo_id: Optional[str] = None
    vorp: Optional[float] = None

    class Config:
        use_enum_values = True

    @classmethod
    def from_players(cls, player: Player) -> 'PlayerAPI':
        return cls.model_validate(player)