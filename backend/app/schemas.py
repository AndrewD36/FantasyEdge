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
    current_team_abbr: Optional[str] = None
 
 
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


class GameRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    game_id: str
    season: int
    week: int
    game_type: str
    gameday: Optional[date] = None
    weekday: Optional[str] = None
    gametime: Optional[str] = None

    away_team: str
    home_team: str
    away_score: Optional[int] = None
    home_score: Optional[int] = None
    result: Optional[int] = None
    total: Optional[int] = None
    overtime: bool = False
    div_game: bool = False

    roof: Optional[str] = None
    surface: Optional[str] = None
    temp: Optional[int] = None
    wind: Optional[int] = None

    away_qb_id: Optional[str] = None
    home_qb_id: Optional[str] = None
    away_qb_name: Optional[str] = None
    home_qb_name: Optional[str] = None
    away_coach: Optional[str] = None
    home_coach: Optional[str] = None
    referee: Optional[str] = None
    stadium: Optional[str] = None

    spread_line: Optional[float] = None
    total_line: Optional[float] = None


class SnapCountBase(BaseModel):
    pfr_player_id: str
    game_id: str
    season: int
    week: int
    game_type: str
    player_name: str
    position: Optional[str] = None
    team_abbr: Optional[str] = None
    opponent_abbr: Optional[str] = None

    offense_snaps: Optional[float] = None
    offense_pct: Optional[float] = None
    defense_snaps: Optional[float] = None
    defense_pct: Optional[float] = None
    st_snaps: Optional[float] = None
    st_pct: Optional[float] = None


class SnapCountCreate(SnapCountBase):
    player_id: Optional[str] = None  # nullable — resolved via pfr_id, not always found


class SnapCountRead(SnapCountBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: Optional[str] = None


class NgsPassingBase(BaseModel):
    season: int
    week: int
    season_type: str
    team_abbr: Optional[str] = None

    avg_time_to_throw: Optional[float] = None
    avg_completed_air_yards: Optional[float] = None
    avg_intended_air_yards: Optional[float] = None
    avg_air_yards_differential: Optional[float] = None
    aggressiveness: Optional[float] = None
    max_completed_air_distance: Optional[float] = None
    avg_air_yards_to_sticks: Optional[float] = None
    completion_percentage: Optional[float] = None
    expected_completion_percentage: Optional[float] = None
    completion_percentage_above_expectation: Optional[float] = None
    avg_air_distance: Optional[float] = None
    max_air_distance: Optional[float] = None
    passer_rating: Optional[float] = None

    @field_validator("season_type")
    @classmethod
    def valid_season_type(cls, v: str) -> str:
        allowed = {"REG", "POST"}
        if v not in allowed:
            raise ValueError(f"season_type must be one of {allowed}")
        return v


class NgsPassingCreate(NgsPassingBase):
    player_id: str


class NgsPassingRead(NgsPassingBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: str


class NgsReceivingBase(BaseModel):
    season: int
    week: int
    season_type: str
    team_abbr: Optional[str] = None

    avg_cushion: Optional[float] = None
    avg_separation: Optional[float] = None
    avg_intended_air_yards: Optional[float] = None
    percent_share_of_intended_air_yards: Optional[float] = None
    catch_percentage: Optional[float] = None
    avg_yac: Optional[float] = None
    avg_expected_yac: Optional[float] = None
    avg_yac_above_expectation: Optional[float] = None

    @field_validator("season_type")
    @classmethod
    def valid_season_type(cls, v: str) -> str:
        allowed = {"REG", "POST"}
        if v not in allowed:
            raise ValueError(f"season_type must be one of {allowed}")
        return v


class NgsReceivingCreate(NgsReceivingBase):
    player_id: str


class NgsReceivingRead(NgsReceivingBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: str


class NgsRushingBase(BaseModel):
    season: int
    week: int
    season_type: str
    team_abbr: Optional[str] = None

    efficiency: Optional[float] = None
    percent_attempts_gte_eight_defenders: Optional[float] = None
    avg_time_to_los: Optional[float] = None
    avg_rush_yards: Optional[float] = None
    expected_rush_yards: Optional[float] = None
    rush_yards_over_expected: Optional[float] = None
    rush_yards_over_expected_per_att: Optional[float] = None
    rush_pct_over_expected: Optional[float] = None

    @field_validator("season_type")
    @classmethod
    def valid_season_type(cls, v: str) -> str:
        allowed = {"REG", "POST"}
        if v not in allowed:
            raise ValueError(f"season_type must be one of {allowed}")
        return v


class NgsRushingCreate(NgsRushingBase):
    player_id: str


class NgsRushingRead(NgsRushingBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: str