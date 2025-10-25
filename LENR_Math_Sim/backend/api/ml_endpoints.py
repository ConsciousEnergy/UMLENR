"""ML-powered API endpoints for parameter optimization."""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from datetime import datetime
import logging

from ml.parameter_optimizer import LENRParameterOptimizer, PatternRecognizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["machine-learning"])

# Global ML instances
optimizer = LENRParameterOptimizer()
recognizer = PatternRecognizer()


@router.post("/optimize/suggest")
async def suggest_parameters(exploration_weight: float = 1.0) -> Dict[str, Any]:
    """
    Get ML-suggested parameters for next simulation.
    Uses Bayesian optimization with Gaussian Process.
    """
    try:
        result = optimizer.suggest_next_parameters(exploration_weight=exploration_weight)
        
        return {
            "suggested_parameters": result.optimal_params,
            "expected_enhancement": result.expected_enhancement,
            "uncertainty": result.uncertainty,
            "acquisition_value": result.acquisition_value,
            "n_observations": len(optimizer.X_train)
        }
    except Exception as e:
        logger.error(f"Parameter suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/observe")
async def add_observation(params: Dict[str, float], enhancement: float) -> Dict[str, str]:
    """
    Add a new observation to train the ML model.
    """
    try:
        optimizer.add_observation(params, enhancement)
        
        # Retrain if enough data
        if len(optimizer.X_train) >= 2:
            optimizer.train()
            
        return {
            "status": "success",
            "message": f"Added observation #{len(optimizer.X_train)}"
        }
    except Exception as e:
        logger.error(f"Failed to add observation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns/analyze")
async def analyze_patterns(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze simulation data for patterns in high-yield conditions.
    """
    try:
        patterns = recognizer.analyze_patterns(data)
        return patterns
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimize/best")
async def get_best_parameters() -> Dict[str, Any]:
    """
    Get the best parameters found so far by the optimizer.
    """
    try:
        best_params = optimizer.get_best_parameters()
        
        # Predict enhancement for best params
        if len(optimizer.X_train) >= 2:
            enhancement, uncertainty = optimizer.predict(best_params)
        else:
            enhancement, uncertainty = 1e6, 1e6
        
        return {
            "best_parameters": best_params,
            "expected_enhancement": enhancement,
            "uncertainty": uncertainty,
            "n_observations": len(optimizer.X_train)
        }
    except Exception as e:
        logger.error(f"Failed to get best parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))
