from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

class TeamSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
 
    team_abbr: str = Field(max_length=3)
    team_name: str
    conference: Optional[str] = None
    division: Optional[str] = None
 
 
class PlayerBase(BaseModel):
    full_name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    position: Optional[str] = None
    birth_date: Optional[date] = None
    height: Optional[int] = None
    weight: Optional[int] = None
    college: Optional[str] = None
    draft_year: Optional[int] = None
    draft_round: Optional[int] = None
    draft_pick: Optional[int] = None
    rookie_season: Optional[int] = None
    status: Optional[str] = None
    espn_id: Optional[str] = None
    sleeper_id: Optional[str] = None
    yahoo_id: Optional[str] = None
 
    @field_validator("position")
    @classmethod
    def upper_position(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v else v
 
 
class PlayerCreate(PlayerBase):
    """Used when upserting a row from nflreadpy.load_players()."""
 
    player_id: str  # gsis_id — required on create, nflreadpy always provides it
 
 
class PlayerRead(PlayerBase):
    model_config = ConfigDict(from_attributes=True)
 
    player_id: str
    updated_at: datetime
 
 
class PlayerTeamSeasonSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
 
    player_id: str
    season: int
    team_abbr: str
    week: int = 0
 
 
class PlayerStatBase(BaseModel):
    season: int
    week: int
    season_type: str = "REG"
    team_abbr: Optional[str] = None
    opponent_abbr: Optional[str] = None
 
    completions: int = 0
    attempts: int = 0
    passing_yards: float = 0
    passing_tds: int = 0
    interceptions: int = 0
    sacks: int = 0
    sack_yards: float = 0
 
    carries: int = 0
    rushing_yards: float = 0
    rushing_tds: int = 0
    rushing_fumbles: int = 0
 
    targets: int = 0
    receptions: int = 0
    receiving_yards: float = 0
    receiving_tds: int = 0
    receiving_fumbles: int = 0
 
    fantasy_points: Optional[float] = None
    fantasy_points_ppr: Optional[float] = None
 
    @field_validator("season_type")
    @classmethod
    def valid_season_type(cls, v: str) -> str:
        allowed = {"REG", "POST", "PRE"}
        if v not in allowed:
            raise ValueError(f"season_type must be one of {allowed}")
        return v
 
 
class PlayerStatCreate(PlayerStatBase):
    player_id: str
 
 
class PlayerStatRead(PlayerStatBase):
    model_config = ConfigDict(from_attributes=True)
 
    id: int
    player_id: str
 
 
class PlayerWithStatsRead(PlayerRead):
    """Nested read schema — handy for a 'player detail' API response
    that includes their season stat log in one payload."""
 
    stats: list[PlayerStatRead] = []