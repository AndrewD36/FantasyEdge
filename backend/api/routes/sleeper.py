"""
Sleeper API integration endpoints.

Handles all interactions with Sleeper's API including user authentication,
league discovery, and real-time draft data synchronization.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from ...external.sleeper_client import SleeperClient
from ...datamodels.draft_state import DraftState, DraftStateResponse


router = APIRouter()


class UserInfo(BaseModel):
    user_id: str
    username: str
    display_name: Optional[str]
    avatar: Optional[str]


class LeagueInfo(BaseModel):
    league_id: str
    name: str
    season: str
    total_rosters: int
    status: str  # "pre_draft", "drafting", "in_season", "complete"
    draft_id: Optional[str]


class ConnectSleeperResponse(BaseModel):
    user_info: UserInfo
    leagues: List[LeagueInfo]
    active_drafts: List[str]

async def get_sleeper_client() -> SleeperClient:
    """
    Dependency to provide Sleeper API client.
    
    In the future, this could handle connection pooling,
    rate limiting, authentication, etc.
    """
    return SleeperClient()

@router.post("/sleeper/connect/{username}", response_model=ConnectSleeperResponse)
async def connect_sleeper_account(
    username: str,
    sleeper_client: SleeperClient = Depends(get_sleeper_client)
):
    """
    Connect to a user's Sleeper account and discover their leagues.
    
    This is the entry point for users to connect their Sleeper data.
    Returns user info, leagues, and any active drafts.
    """
    
    try:
        # Get user information
        user_data = await sleeper_client.get_user(username)
        if not user_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Sleeper user '{username}' not found"
            )
        
        user_info = UserInfo(
            user_id=user_data['user_id'],
            username=user_data['username'],
            display_name=user_data.get('display_name'),
            avatar=user_data.get('avatar')
        )
        
        # Get user's leagues for current season
        leagues_data = await sleeper_client.get_user_leagues(user_data['user_id'])
        
        leagues = []
        active_drafts = []
        
        for league_data in leagues_data:
            league_info = LeagueInfo(
                league_id=league_data['league_id'],
                name=league_data['name'],
                season=league_data['season'],
                total_rosters=league_data['total_rosters'],
                status=league_data['status'],
                draft_id=league_data.get('draft_id')
            )
            leagues.append(league_info)
            
            # Track active drafts
            if league_data['status'] == 'drafting' and league_data.get('draft_id'):
                active_drafts.append(league_data['draft_id'])
        
        return ConnectSleeperResponse(
            user_info=user_info,
            leagues=leagues,
            active_drafts=active_drafts
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error connecting to Sleeper: {str(e)}"
        )


@router.get("/sleeper/draft/{draft_id}/state", response_model=DraftStateResponse)
async def get_draft_state(
    draft_id: str,
    user_team_id: Optional[str] = Query(None, description="User's team ID in the draft"),
    sleeper_client: SleeperClient = Depends(get_sleeper_client)
):
    """
    Get the current state of a Sleeper draft.
    
    Fetches real-time draft data and converts it to our internal format.
    This is called frequently during active drafts.
    """
    
    try:
        # Get draft information and picks
        draft_info, draft_picks = await sleeper_client.get_draft_data(draft_id)
        
        if not draft_info:
            raise HTTPException(
                status_code=404,
                detail=f"Draft '{draft_id}' not found"
            )
        
        # Convert to our internal DraftState format
        draft_state = await sleeper_client.convert_to_draft_state(
            draft_info, 
            draft_picks, 
            user_team_id
        )
        
        # Calculate derived information
        available_players_count = 500 - len(draft_state.drafted_players)  # Rough estimate
        user_turn = draft_state.is_user_turn()
        picks_until_user_turn = draft_state.picks_until_user_turn()
        
        return DraftStateResponse(
            draft_state=draft_state,
            available_players_count=available_players_count,
            user_turn=user_turn,
            picks_until_user_turn=picks_until_user_turn
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching draft state: {str(e)}"
        )


@router.get("/sleeper/league/{league_id}/info", response_model=LeagueInfo)
async def get_league_info(
    league_id: str,
    sleeper_client: SleeperClient = Depends(get_sleeper_client)
):
    """
    Get detailed information about a specific league.
    
    Useful for displaying league settings and context.
    """
    
    try:
        league_data = await sleeper_client.get_league_info(league_id)
        
        if not league_data:
            raise HTTPException(
                status_code=404,
                detail=f"League '{league_id}' not found"
            )
        
        return LeagueInfo(
            league_id=league_data['league_id'],
            name=league_data['name'],
            season=league_data['season'],
            total_rosters=league_data['total_rosters'],
            status=league_data['status'],
            draft_id=league_data.get('draft_id')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching league info: {str(e)}"
        )


@router.post("/sleeper/draft/{draft_id}/sync")
async def sync_draft_state(
    draft_id: str,
    sleeper_client: SleeperClient = Depends(get_sleeper_client)
):
    """
    Force a sync of draft state from Sleeper.
    
    Useful for manual refresh or when real-time updates fail.
    Returns a simple success/failure status.
    """
    
    try:
        # Force refresh draft data
        success = await sleeper_client.sync_draft(draft_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to sync draft state"
            )
        
        return {
            "success": True,
            "message": f"Draft {draft_id} synced successfully",
            "timestamp": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error syncing draft: {str(e)}"
        )


@router.get("/sleeper/players/search")
async def search_sleeper_players(
    query: str = Query(..., min_length=2, description="Player name search query"),
    position: Optional[str] = Query(None, description="Filter by position"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    sleeper_client: SleeperClient = Depends(get_sleeper_client)
):
    """
    Search for players in Sleeper's database.
    
    Useful for player lookup and validation.
    """
    
    try:
        players = await sleeper_client.search_players(
            query=query,
            position=position,
            limit=limit
        )
        
        return {
            "query": query,
            "position_filter": position,
            "results": players,
            "count": len(players)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching players: {str(e)}"
        )