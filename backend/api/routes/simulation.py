"""
Monte Carlo simulation API endpoints.

Provides RESTful access to simulation capabilities with proper
error handling, caching, and performance optimization.
"""

import asyncio
import time
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse

from ...datamodels.simulation import (
    SimulationRequest, SimulationResponse, SimulationParameters,
    SimulationStrategy
)
from ...datamodels.draft_state import DraftState
from ...simulation.monte_carlo import MonteCarloSimulator
from ...external.sleeper_client import SleeperClient


router = APIRouter()
logger = logging.getLogger(__name__)


# Global simulator instance (in production, this would be dependency injected)
_simulator_instance: Optional[MonteCarloSimulator] = None


async def get_simulator() -> MonteCarloSimulator:
    """
    Dependency to get Monte Carlo simulator instance.
    
    In production, this would handle:
    - Connection pooling
    - Model loading
    - Configuration management
    - Health checking
    """
    global _simulator_instance
    
    if _simulator_instance is None:
        # TODO: Load player database
        players_db = {}  # Would load from database/cache
        _simulator_instance = MonteCarloSimulator(players_db)
    
    return _simulator_instance


async def get_sleeper_client() -> SleeperClient:
    """Dependency to get Sleeper API client."""
    return SleeperClient()


@router.post("/simulation/run", response_model=SimulationResponse)
async def run_monte_carlo_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    simulator: MonteCarloSimulator = Depends(get_simulator),
    sleeper_client: SleeperClient = Depends(get_sleeper_client)
):
    """
    Run Monte Carlo simulation for draft availability prediction.
    
    This is the main endpoint that powers the AI recommendations.
    Handles caching, performance optimization, and error recovery.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting simulation for draft {request.draft_id}")
        
        # Get current draft state
        draft_info, draft_picks = await sleeper_client.get_draft_data(request.draft_id)
        
        if not draft_info:
            raise HTTPException(
                status_code=404,
                detail=f"Draft {request.draft_id} not found"
            )
        
        draft_state = await sleeper_client.convert_to_draft_state(
            draft_info, draft_picks
        )
        
        # Use provided parameters or defaults
        parameters = request.parameters or SimulationParameters()
        
        # Check cache first (TODO: implement caching)
        if request.use_cache:
            cached_result = await _check_simulation_cache(request, draft_state)
            if cached_result:
                api_time = (time.time() - start_time) * 1000
                return SimulationResponse(
                    success=True,
                    result=cached_result,
                    cached=True,
                    api_response_time_ms=api_time,
                    cache_hit=True
                )
        
        # Run simulation
        simulation_result = await simulator.run_simulation(
            draft_state=draft_state,
            parameters=parameters,
            target_players=request.target_players
        )
        
        # Cache result for future use
        if request.use_cache:
            background_tasks.add_task(
                _cache_simulation_result,
                request,
                draft_state,
                simulation_result
            )
        
        # Calculate API response time
        api_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Simulation completed for draft {request.draft_id} in {api_time:.1f}ms"
        )
        
        return SimulationResponse(
            success=True,
            result=simulation_result,
            cached=False,
            api_response_time_ms=api_time,
            cache_hit=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation failed for draft {request.draft_id}: {e}")
        
        api_time = (time.time() - start_time) * 1000
        
        return SimulationResponse(
            success=False,
            result=None,
            cached=False,
            error_message=f"Simulation failed: {str(e)}",
            api_response_time_ms=api_time,
            cache_hit=False
        )


@router.get("/simulation/quick/{draft_id}")
async def quick_simulation(
    draft_id: str,
    num_simulations: int = Query(200, ge=50, le=1000),
    strategy: SimulationStrategy = Query(SimulationStrategy.BALANCED),
    simulator: MonteCarloSimulator = Depends(get_simulator),
    sleeper_client: SleeperClient = Depends(get_sleeper_client)
):
    """
    Quick simulation endpoint for real-time updates.
    
    Optimized for speed with fewer simulations and simplified parameters.
    Used for live draft updates where sub-second response is critical.
    """
    try:
        # Get draft state
        draft_info, draft_picks = await sleeper_client.get_draft_data(draft_id)
        
        if not draft_info:
            raise HTTPException(status_code=404, detail="Draft not found")
        
        draft_state = await sleeper_client.convert_to_draft_state(
            draft_info, draft_picks
        )
        
        # Quick simulation parameters
        parameters = SimulationParameters(
            num_simulations=num_simulations,
            strategy=strategy,
            adp_adherence=0.7,
            need_weight=0.3,
            chaos_factor=0.1
        )
        
        # Run simulation with timeout
        result = await asyncio.wait_for(
            simulator.run_simulation(draft_state, parameters),
            timeout=5.0  # 5-second timeout for quick simulation
        )
        
        # Return simplified response
        return {
            "draft_id": draft_id,
            "simulation_id": result.simulation_id,
            "execution_time_ms": result.execution_time_ms,
            "players_analyzed": result.total_players_analyzed,
            "urgent_picks": len(result.get_urgent_picks()),
            "safe_picks": len(result.get_safe_picks()),
            "top_availabilities": {
                player_id: availability.availability_probability
                for player_id, availability in list(result.player_availabilities.items())[:10]
            }
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail="Simulation timed out - try reducing num_simulations"
        )
    except Exception as e:
        logger.error(f"Quick simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation/player/{player_id}/availability")
async def get_player_availability(
    player_id: str,
    draft_id: str = Query(...),
    simulator: MonteCarloSimulator = Depends(get_simulator),
    sleeper_client: SleeperClient = Depends(get_sleeper_client)
):
    """
    Get availability prediction for a specific player.
    
    Focused endpoint for single-player analysis, useful for
    detailed player evaluation and comparison.
    """
    try:
        # Get draft state
        draft_info, draft_picks = await sleeper_client.get_draft_data(draft_id)
        draft_state = await sleeper_client.convert_to_draft_state(draft_info, draft_picks)
        
        # Check if player already drafted
        if player_id in draft_state.drafted_players:
            return {
                "player_id": player_id,
                "already_drafted": True,
                "drafted_at_pick": next(
                    (pick.pick_number for pick in draft_state.completed_picks 
                     if pick.player_id == player_id), None
                )
            }
        
        # Run targeted simulation
        parameters = SimulationParameters(num_simulations=500)
        result = await simulator.run_simulation(
            draft_state, parameters, target_players=[player_id]
        )
        
        availability = result.get_player_availability(player_id)
        
        if not availability:
            raise HTTPException(status_code=404, detail="Player not found in simulation")
        
        return {
            "player_id": player_id,
            "availability_probability": availability.availability_probability,
            "urgency_score": availability.urgency_score,
            "average_pick_taken": availability.average_pick_taken,
            "risk_assessment": availability.risk_assessment,
            "competing_teams": availability.competing_teams,
            "simulation_id": availability.simulation_id
        }
        
    except Exception as e:
        logger.error(f"Player availability failed for {player_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation/insights/{draft_id}")
async def get_draft_insights(
    draft_id: str,
    simulator: MonteCarloSimulator = Depends(get_simulator),
    sleeper_client: SleeperClient = Depends(get_sleeper_client)
):
    """
    Get high-level draft insights and market analysis.
    
    Provides strategic insights about draft flow, position runs,
    and market opportunities without running full simulation.
    """
    try:
        # Get draft state
        draft_info, draft_picks = await sleeper_client.get_draft_data(draft_id)
        draft_state = await sleeper_client.convert_to_draft_state(draft_info, draft_picks)
        
        # Quick simulation for insights
        parameters = SimulationParameters(num_simulations=300)
        result = await simulator.run_simulation(draft_state, parameters)
        
        return {
            "draft_id": draft_id,
            "current_pick": draft_state.current_pick,
            "completion_percentage": draft_state.completion_percentage,
            "position_scarcity": result.position_scarcity,
            "likely_runs": result.likely_runs,
            "value_opportunities": result.value_opportunities,
            "market_temperature": _calculate_market_temperature(result),
            "recommendations": _generate_strategic_recommendations(result, draft_state)
        }
        
    except Exception as e:
        logger.error(f"Draft insights failed for {draft_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for caching and analysis
async def _check_simulation_cache(request: SimulationRequest, 
                                draft_state: DraftState) -> Optional[any]:
    """Check if simulation result is cached."""
    # TODO: Implement Redis-based caching
    # Cache key would be based on draft state + simulation parameters
    return None


async def _cache_simulation_result(request: SimulationRequest,
                                 draft_state: DraftState,
                                 result: any):
    """Cache simulation result for future use."""
    # TODO: Implement Redis caching with appropriate TTL
    pass


def _calculate_market_temperature(result: any) -> str:
    """
    Calculate overall market temperature.
    
    Hot = lots of urgency, Cold = many safe picks available
    """
    urgent_count = len(result.get_urgent_picks())
    safe_count = len(result.get_safe_picks())
    total = result.total_players_analyzed
    
    if total == 0:
        return "neutral"
    
    urgency_ratio = urgent_count / total
    
    if urgency_ratio > 0.3:
        return "hot"
    elif urgency_ratio < 0.1:
        return "cold"
    else:
        return "neutral"


def _generate_strategic_recommendations(result: any, draft_state: DraftState) -> List[str]:
    """Generate high-level strategic recommendations."""
    recommendations = []
    
    # Position run recommendations
    if result.likely_runs:
        recommendations.append(
            f"Position runs likely in: {', '.join(result.likely_runs)}"
        )
    
    # Value opportunity recommendations
    if result.value_opportunities:
        recommendations.append(
            f"Value opportunities available in {len(result.value_opportunities)} players"
        )
    
    # Urgency recommendations
    urgent_picks = result.get_urgent_picks()
    if urgent_picks:
        recommendations.append(
            f"{len(urgent_picks)} players unlikely to be available next round"
        )
    
    # Default recommendation
    if not recommendations:
        recommendations.append("Draft proceeding normally - focus on best available value")
    
    return recommendations