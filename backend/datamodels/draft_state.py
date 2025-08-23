"""
Draft state management models.

Represents the current state of a fantasy draft, including picks made,
team rosters, and draft configuration. Designed for real-time updates.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from .player import Player, PlayerPosition

class DraftType(str, Enum):
    SNAKE = "snake"
    LINEAR = "linear"
    AUCTION = "auction"

class DraftStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"

class RosterConfig(BaseModel):
    #TODO: add Field properties to each variable below
    qb: int
    rb: int
    wr: int
    te: int
    flex: int
    superflex: int
    k: int
    dst: int
    bench: int


    @field_validator('*', pre=True)
    def roster_value_check(cls, v):
        if isinstance(v, int) and v < 0:
            raise ValueError('Roster spots cannot be negative')
        return v
    
    @property
    def total_roster_spots(self) -> int:
        return (self.qb + self.rb + self.wr + self.te + self.flex + self.superflex + self.k + self.dst + self.bench)
    
    @property
    def starting_spots(self) -> int:
        return (self.qb + self.rb + self.wr + self.te + self.flex + self.superflex + self.k + self.dst)
    
    def position_need(self, position: PlayerPosition, current_count: int) -> float:
        required = getattr(self, position.value.lower(), 0)

        if current_count < required:
            return 1.0 + (required - current_count) * 0.2
        elif position in [PlayerPosition.RB, PlayerPosition.WR, PlayerPosition.TE]:
            flex_need = max(0, self.flex - max(0, current_count - required))
            return 0.3 + (flex_need * 0.1)
        else:
            return 0.1 # arbitrary value for depth
        
class TeamRoster(BaseModel):
    #TODO: add Field properties to each variable below
    team_id: str
    team_name: Optional[str]
    owner_name: Optional[str]

    players: List[str]
    draft_order: List[int]

    qb_count: int
    rb_count: int
    wr_count: int
    te_count: int
    k_count: int
    dst_count: int

    def add_player(self, player_id: str, pick_num: int, position: PlayerPosition) -> int:
        self.players.append(player_id)
        self.draft_order.append(pick_num)

        pos_field = f'{position.value.lower()}_count'
        if hasattr(self, pos_field):
            current_count = getattr(self, pos_field)
            setattr(self, pos_field, current_count + 1)

    def get_position_count(self, position: PlayerPosition) -> int:
        pos_field = f'{position.value.lower()}_count'
        return getattr(self, pos_field, 0)
    
    @property
    def total_players(self) -> int:
        return len(self.players)
    
class DraftPick(BaseModel):
    #TODO: add Field properties to each variable below
    pick_num: int
    round_num: int
    round_pick: int
    team_id: str
    player_id: Optional[str]
    timestamp: Optional[datetime]
    time_on_clock: Optional[float]

class DraftState(BaseModel):
    #TODO: add Field properties to each variable below
    draft_id: str
    league_id: str

    draft_type: DraftType
    status: DraftStatus
    roster_config: RosterConfig

    teams: Dict[str: TeamRoster]
    team_count: int
    rounds: int
    picks_per_round: int
    total_picks: int

    current_pick: int
    current_team_id: Optional[str]

    completed_picks: List[DraftPick]
    drafted_players: Set[str]

    user_team_id: Optional[str]

    created_at: datetime
    started_at: Optional[datetime]
    updated_at: datetime

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat(),
                         set: list}
        
    @field_validator('drafted_players', pre=True)
    def convert_drafted_players_to_set(cls, v):
        if isinstance(v, int):
            return set(v)
        return v
    
    def __init__(self, **data):
        super().__init__(**data)

        if not hasattr(self, 'picks_per_round') or self.picks_per_round is None:
            self.picks_per_round = self.team_count

        if not hasattr(self, 'total_picks') or self.total_picks is None:
            self.total_picks = self.team_count * self.rounds

    def get_current_picking_team(self) -> Optional[str]:
        if self.current_pick > self.total_picks:
            return None
        
        if self.draft_type == DraftType.SNAKE:
            round_number = ((self.current_pick - 1) // self.team_count) + 1
            pick_in_round = ((self.current_pick - 1) % self.team_count) + 1

            if round_number % 2 == 1:
                team_index = pick_in_round - 1
            else:
                team_index = self.team_count - pick_in_round

            team_ids = list(self.teams.keys())
            if 0 <= team_index < len(team_ids):
                return team_ids[team_index]
            
        elif self.draft_type == DraftType.LINEAR:
            pick_in_round = ((self.current_pick - 1) % self.team_count) + 1
            team_index = pick_in_round - 1
            team_ids = list(self.teams.keys())
            if 0 <= team_index < len(team_ids):
                return team_ids[team_index]
            
        return None
    
    def add_pick(self, player_id: str, team_id: str) -> bool:
        if player_id in self.drafted_players:
            return False
        
        if team_id not in self.teams:
            return False
        
        round_number = ((self.current_pick - 1) // self.team_count) + 1
        pick_in_round = ((self.current_pick - 1) % self.team_count) + 1

        pick = DraftPick(pick_num=self.current_pick,
                         round_num=round_number,
                         round_pick=pick_in_round,
                         team_id=team_id,
                         player_id=player_id,
                         timestamp=datetime.now(timezone.utc))
        
        self.completed_picks.append(pick)
        self.drafted_players.add(player_id)
        self.current_pick += 1
        self.updated_at = datetime.now(timezone.utc)

        return True
    
    def get_user_team(self) -> Optional[TeamRoster]:
        if self.user_team_id and self.user_team_id in self.teams:
            return self.teams[self.user_team_id]
        return None
    
    def is_user_turn(self) -> bool:
        current_team = self.get_current_picking_team()
        return current_team == self.user_team_id
    
    def picks_until_user_turn(self) -> int:
        if not self.user_team_id:
            return -1
        
        current_team = self.get_current_picking_team()
        if current_team == self.user_team_id:
            return 0
        
        team_ids = list(self.teams.keys())
        try:
            user_index = team_ids.index(self.user_team_id)
            current_index = team_ids.index(current_team) if current_team else 0

            if user_index > current_index:
                return user_index - current_index
            else:
                return (len(team_ids) - current_index) + user_index
            
        except ValueError:
            return -1
        
    def get_available_players_by_position(self, position: PlayerPosition) -> List[str]:
        """
        Get available player IDs for a specific position.
        
        Optimized for simulation loops where we frequently need
        position-filtered player lists.
        """
        # This would integrate with player database
        # Simplified implementation for now
        return [pid for pid in self.players_db.keys() 
                if (pid not in self.drafted_players and 
                    self.players_db[pid].position == position)]

    def calculate_positional_scarcity(self, position: PlayerPosition) -> float:
        """
        Calculate real-time positional scarcity based on draft progress.
        
        Returns 0.0 (abundant) to 1.0 (very scarce) based on:
        - How many quality players remain at position
        - How fast the position is being drafted
        - Expected positional needs across all teams
        """
        available_at_pos = len(self.get_available_players_by_position(position))
        
        # Calculate expected remaining demand
        teams_needing_position = 0
        for team_roster in self.teams.values():
            team_pos_count = len([p for p in team_roster.players 
                                if self.players_db.get(p, {}).get('position') == position])
            
            required = self.roster_spots.get(position.value.lower(), 0)
            if team_pos_count < required:
                teams_needing_position += 1
        
        # Scarcity = demand / supply
        if available_at_pos == 0:
            return 1.0
        
        scarcity_ratio = teams_needing_position / available_at_pos
        return min(1.0, scarcity_ratio)

    def get_draft_velocity(self, position: PlayerPosition, window: int = 5) -> float:
        """
        Calculate drafting velocity for a position (picks per round).
        
        Used to predict if position runs are starting.
        Higher velocity = position being drafted faster than normal.
        """
        if len(self.completed_picks) < window:
            return 0.0
        
        recent_picks = self.completed_picks[-window:]
        position_picks = sum(1 for pick in recent_picks 
                            if (pick.player_id in self.players_db and 
                                self.players_db[pick.player_id].position == position))
        
        return position_picks / window  # Picks per round velocity
        
    @property
    def is_complete(self) -> bool:
        return self.current_pick > self.total_picks
    
    @property
    def complete_percentage(self) -> float:
        return min(100.0, (len(self.completed_picks) / self.total_picks) * 100)
    
    @property
    def draft_phase(self) -> str:
        """
        Determine current draft phase for behavioral modeling.
        
        Different phases have different drafting psychology:
        - Early: Value-focused, best player available
        - Middle: Balanced value/need approach  
        - Late: Need-focused, filling roster holes
        """
        completion_pct = self.completion_percentage
        
        if completion_pct < 30:
            return "early"
        elif completion_pct < 70:
            return "middle"
        else:
            return "late"

    @property
    def time_pressure_factor(self) -> float:
        """
        Calculate time pressure factor affecting draft decisions.
        
        Higher pressure leads to more ADP-adherent picks as
        drafters rely on consensus rather than deep analysis.
        """
        # Later in draft = more time pressure (fatigue, less time per pick)
        base_pressure = self.completion_percentage / 100.0
        
        # Position runs create additional pressure
        recent_velocity = sum(self.get_draft_velocity(pos) for pos in PlayerPosition) / len(PlayerPosition)
        velocity_pressure = min(0.3, recent_velocity * 0.5)
        
        return min(1.0, base_pressure + velocity_pressure)
    
class DraftStateResponse(BaseModel):
    draft_state: DraftState
    available_players_count: int
    user_turn: bool
    picks_until_user_turn: int

    class Config:
        use_enum_values = True