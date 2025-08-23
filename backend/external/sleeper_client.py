"""
Sleeper API client for fantasy football data.

Handles all communication with Sleeper's REST API including
rate limiting, error handling, and data transformation.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
import httpx
import logging
from datetime import datetime

from ..datamodels.draft_state import DraftState, TeamRoster, RosterConfiguration, DraftPick
from ..datamodels.player import PlayerPosition


logger = logging.getLogger(__name__)


class SleeperAPIError(Exception):
    """Custom exception for Sleeper API errors."""
    pass


class SleeperRateLimitError(SleeperAPIError):
    """Raised when hitting Sleeper API rate limits."""
    pass


class SleeperClient:
    """
    Async client for Sleeper API with rate limiting and error handling.
    
    Sleeper's API is generally reliable but has rate limits.
    This client handles retries, backoff, and data transformation.
    """
    
    BASE_URL = "https://api.sleeper.app/v1"
    
    def __init__(self, 
                 timeout: float = 10.0,
                 max_retries: int = 3,
                 rate_limit_delay: float = 1.0):
        """
        Initialize Sleeper API client.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            rate_limit_delay: Delay between requests to respect rate limits
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
        
        # HTTP client with sensible defaults
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "User-Agent": "FantasyDraftAI/1.0.0",
                "Accept": "application/json",
            },
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def _make_request(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make a request to Sleeper API with rate limiting and retries.
        
        Args:
            endpoint: API endpoint (e.g., "/user/username")
            **kwargs: Additional arguments for httpx.get()
        
        Returns:
            JSON response as dictionary
        
        Raises:
            SleeperAPIError: For API errors
            SleeperRateLimitError: For rate limit errors
        """
        
        # Simple rate limiting
        now = time.time()
        time_since_last = now - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                self._last_request_time = time.time()
                
                response = await self.client.get(url, **kwargs)
                
                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < self.max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise SleeperRateLimitError("Rate limit exceeded")
                
                # Handle other errors
                if response.status_code == 404:
                    return None  # Not found is often expected
                
                response.raise_for_status()
                
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if attempt < self.max_retries and e.response.status_code >= 500:
                    # Retry on server errors
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error {e.response.status_code}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise SleeperAPIError(f"HTTP {e.response.status_code}: {e.response.text}")
            
            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request error {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise SleeperAPIError(f"Request failed: {e}")
        
        raise SleeperAPIError("Max retries exceeded")
    
    async def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user information by username.
        
        Args:
            username: Sleeper username
            
        Returns:
            User data dictionary or None if not found
        """
        logger.info(f"Fetching user data for: {username}")
        
        try:
            user_data = await self._make_request(f"/user/{username}")
            
            if user_data:
                logger.info(f"Found user: {user_data.get('username')} (ID: {user_data.get('user_id')})")
            
            return user_data
            
        except SleeperAPIError as e:
            logger.error(f"Error fetching user {username}: {e}")
            raise
    
    async def get_user_leagues(self, user_id: str, season: str = "2024") -> List[Dict[str, Any]]:
        """
        Get all leagues for a user in a specific season.
        
        Args:
            user_id: Sleeper user ID
            season: Season year (defaults to current season)
            
        Returns:
            List of league data dictionaries
        """
        logger.info(f"Fetching leagues for user {user_id}, season {season}")
        
        try:
            leagues_data = await self._make_request(f"/user/{user_id}/leagues/nfl/{season}")
            
            if not leagues_data:
                leagues_data = []
            
            logger.info(f"Found {len(leagues_data)} leagues for user {user_id}")
            return leagues_data
            
        except SleeperAPIError as e:
            logger.error(f"Error fetching leagues for user {user_id}: {e}")
            raise
    
    async def get_league_info(self, league_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific league.
        
        Args:
            league_id: Sleeper league ID
            
        Returns:
            League data dictionary or None if not found
        """
        logger.info(f"Fetching league info for: {league_id}")
        
        try:
            league_data = await self._make_request(f"/league/{league_id}")
            
            if league_data:
                logger.info(f"Found league: {league_data.get('name')} ({league_data.get('status')})")
            
            return league_data
            
        except SleeperAPIError as e:
            logger.error(f"Error fetching league {league_id}: {e}")
            raise
    
    async def get_draft_info(self, draft_id: str) -> Optional[Dict[str, Any]]:
        """
        Get draft configuration and metadata.
        
        Args:
            draft_id: Sleeper draft ID
            
        Returns:
            Draft info dictionary or None if not found
        """
        logger.info(f"Fetching draft info for: {draft_id}")
        
        try:
            draft_data = await self._make_request(f"/draft/{draft_id}")
            
            if draft_data:
                logger.info(f"Found draft: {draft_data.get('status')} - {draft_data.get('type')}")
            
            return draft_data
            
        except SleeperAPIError as e:
            logger.error(f"Error fetching draft {draft_id}: {e}")
            raise
    
    async def get_draft_picks(self, draft_id: str) -> List[Dict[str, Any]]:
        """
        Get all picks made in a draft.
        
        Args:
            draft_id: Sleeper draft ID
            
        Returns:
            List of pick data dictionaries
        """
        logger.info(f"Fetching draft picks for: {draft_id}")
        
        try:
            picks_data = await self._make_request(f"/draft/{draft_id}/picks")
            
            if not picks_data:
                picks_data = []
            
            completed_picks = [pick for pick in picks_data if pick.get('player_id')]
            logger.info(f"Found {len(completed_picks)} completed picks out of {len(picks_data)} total")
            
            return picks_data
            
        except SleeperAPIError as e:
            logger.error(f"Error fetching picks for draft {draft_id}: {e}")
            raise
    
    async def get_draft_data(self, draft_id: str) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Get both draft info and picks in a single call.
        
        Args:
            draft_id: Sleeper draft ID
            
        Returns:
            Tuple of (draft_info, draft_picks)
        """
        logger.info(f"Fetching complete draft data for: {draft_id}")
        
        try:
            # Make both requests concurrently
            draft_info_task = self.get_draft_info(draft_id)
            draft_picks_task = self.get_draft_picks(draft_id)
            
            draft_info, draft_picks = await asyncio.gather(
                draft_info_task, 
                draft_picks_task
            )
            
            return draft_info, draft_picks
            
        except SleeperAPIError as e:
            logger.error(f"Error fetching draft data for {draft_id}: {e}")
            raise
    
    async def get_players(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all NFL players from Sleeper.
        
        This is a large response (~2MB) so should be cached.
        
        Returns:
            Dictionary mapping player IDs to player data
        """
        logger.info("Fetching all NFL players from Sleeper")
        
        try:
            players_data = await self._make_request("/players/nfl")
            
            if not players_data:
                players_data = {}
            
            logger.info(f"Found {len(players_data)} players")
            return players_data
            
        except SleeperAPIError as e:
            logger.error(f"Error fetching players: {e}")
            raise
    
    async def search_players(self, 
                           query: str, 
                           position: Optional[str] = None,
                           limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for players by name.
        
        Note: Sleeper doesn't have a search endpoint, so this fetches
        all players and filters client-side. In production, you'd
        want to cache the player data.
        
        Args:
            query: Player name search query
            position: Optional position filter
            limit: Maximum number of results
            
        Returns:
            List of matching player dictionaries
        """
        logger.info(f"Searching players: '{query}' (position: {position})")
        
        try:
            # TODO: In production, cache this data
            all_players = await self.get_players()
            
            results = []
            query_lower = query.lower()
            
            for player_id, player_data in all_players.items():
                # Skip if no name
                if not player_data.get('full_name'):
                    continue
                
                # Check name match
                if query_lower not in player_data['full_name'].lower():
                    continue
                
                # Check position filter
                if position and player_data.get('position') != position.upper():
                    continue
                
                # Add player ID to result
                player_data['player_id'] = player_id
                results.append(player_data)
                
                # Limit results
                if len(results) >= limit:
                    break
            
            logger.info(f"Found {len(results)} matching players")
            return results
            
        except SleeperAPIError as e:
            logger.error(f"Error searching players: {e}")
            raise
    
    async def convert_to_draft_state(self, 
                                   draft_info: Dict[str, Any],
                                   draft_picks: List[Dict[str, Any]],
                                   user_team_id: Optional[str] = None) -> DraftState:
        """
        Convert Sleeper draft data to our internal DraftState format.
        
        This is a key function that transforms external data into our
        standardized internal representation.
        
        Args:
            draft_info: Sleeper draft information
            draft_picks: List of Sleeper draft picks
            user_team_id: Optional user team ID
            
        Returns:
            DraftState object
        """
        logger.info(f"Converting Sleeper draft data to internal format")
        
        try:
            # Extract basic draft info
            draft_id = draft_info['draft_id']
            league_id = draft_info['league_id']
            team_count = draft_info['settings']['teams']
            rounds = draft_info['settings']['rounds']
            
            # Create roster configuration from draft settings
            settings = draft_info['settings']
            roster_config = RosterConfiguration(
                qb=settings.get('qb', 1),
                rb=settings.get('rb', 2),
                wr=settings.get('wr', 2),
                te=settings.get('te', 1),
                flex=settings.get('flex', 1),
                superflex=settings.get('super_flex', 0),  # Note: Sleeper uses 'super_flex'
                k=settings.get('k', 1),
                dst=settings.get('def', 1),  # Note: Sleeper uses 'def'
                bench=settings.get('bn', 6)  # Note: Sleeper uses 'bn'
            )
            
            # Create team rosters
            teams = {}
            for i in range(1, team_count + 1):
                team_id = str(i)
                teams[team_id] = TeamRoster(
                    team_id=team_id,
                    team_name=f"Team {i}",  # TODO: Get actual team names
                    owner_name=None  # TODO: Get owner names
                )
            
            # Process completed picks
            completed_picks = []
            drafted_players = set()
            current_pick = 1
            
            for pick_data in draft_picks:
                if pick_data.get('player_id'):
                    # Convert Sleeper pick to our format
                    pick = DraftPick(
                        pick_number=pick_data['pick_no'],
                        round_number=pick_data['round'],
                        pick_in_round=pick_data['draft_slot'],
                        team_id=str(pick_data['roster_id']),
                        player_id=pick_data['player_id'],
                        timestamp=datetime.fromtimestamp(pick_data['picked_at'] / 1000) if pick_data.get('picked_at') else None
                    )
                    
                    completed_picks.append(pick)
                    drafted_players.add(pick_data['player_id'])
                    
                    # Add player to team roster
                    team_id = str(pick_data['roster_id'])
                    if team_id in teams:
                        # TODO: Get actual position from player data
                        # For now, we'll update this when we have player lookup
                        teams[team_id].add_player(
                            pick_data['player_id'],
                            pick_data['pick_no'],
                            PlayerPosition.QB  # Placeholder
                        )
                    
                    current_pick = max(current_pick, pick_data['pick_no'] + 1)
            
            # Determine draft status
            if draft_info['status'] == 'complete':
                status = "completed"
            elif draft_info['status'] == 'drafting':
                status = "in_progress"
            else:
                status = "not_started"
            
            # Create DraftState
            draft_state = DraftState(
                draft_id=draft_id,
                league_id=league_id,
                draft_type="snake" if draft_info['type'] == 1 else "linear",
                status=status,
                roster_config=roster_config,
                teams=teams,
                team_count=team_count,
                rounds=rounds,
                current_pick=current_pick,
                completed_picks=completed_picks,
                drafted_players=drafted_players,
                user_team_id=user_team_id,
                picks_per_round=team_count,
                total_picks=team_count * rounds
            )
            
            logger.info(f"Successfully converted draft data: {len(completed_picks)} picks, current pick {current_pick}")
            return draft_state
            
        except Exception as e:
            logger.error(f"Error converting draft data: {e}")
            raise SleeperAPIError(f"Failed to convert draft data: {e}")
    
    async def sync_draft(self, draft_id: str) -> bool:
        """
        Force sync of draft data.
        
        This is a placeholder for future caching/sync logic.
        
        Args:
            draft_id: Draft to sync
            
        Returns:
            True if sync successful
        """
        try:
            # For now, just verify we can fetch the data
            draft_info, draft_picks = await self.get_draft_data(draft_id)
            return draft_info is not None
            
        except SleeperAPIError:
            return False


# Example usage and testing
async def example_usage():
    """Example of how to use the SleeperClient."""
    
    async with SleeperClient() as client:
        try:
            # Get user information
            user_data = await client.get_user("your_username")
            if user_data:
                print(f"Found user: {user_data['username']}")
                
                # Get their leagues
                leagues = await client.get_user_leagues(user_data['user_id'])
                print(f"Found {len(leagues)} leagues")
                
                # Look for active drafts
                for league in leagues:
                    if league['status'] == 'drafting' and league.get('draft_id'):
                        draft_id = league['draft_id']
                        print(f"Found active draft: {draft_id}")
                        
                        # Get draft data
                        draft_info, picks = await client.get_draft_data(draft_id)
                        if draft_info:
                            draft_state = await client.convert_to_draft_state(draft_info, picks)
                            print(f"Draft state: {draft_state.completion_percentage:.1f}% complete")
            
        except SleeperAPIError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())