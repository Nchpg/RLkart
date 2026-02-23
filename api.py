from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import uvicorn
import numpy as np
import pybullet as p
import os

from APISimulator import APISimulator
from Car import RLCar
from RLModels import RLModelHandler
from GenTrack import TrackGenerator

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup
MODEL_PATH = "Models/ppo_car"

def make_dummy_env():
    from TestSimulator import TestSimulator
    # Use a real simulator in DIRECT mode for loading spaces
    return TestSimulator(cars_type=[RLCar], mode=p.DIRECT)

model_handler = RLModelHandler(make_dummy_env)
try:
    # This will create one environment instance via make_dummy_env
    model, env = model_handler.load(MODEL_PATH)
    RLCar.set_model((model, env))
    # Disconnect the dummy environment connection
    p.disconnect()
    print(f"Model {MODEL_PATH} loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    try:
        p.disconnect()
    except:
        pass

class ControlPoint(BaseModel):
    x: float
    y: float

class SimulationRequest(BaseModel):
    control_points: List[ControlPoint]
    width: float = 4.0

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    if len(request.control_points) < 3:
        raise HTTPException(status_code=400, detail="At least 3 control points are required")
    
    # Convert Pydantic models to list of points
    pts = [[cp.x, cp.y] for cp in request.control_points]
    
    try:
        # Create a simulator with the custom track
        sim = APISimulator(
            control_points=pts,
            width=4
        )
        
        # Run simulation
        result = sim.run_simulation()
        
        # Cleanup PyBullet
        p.disconnect()
        
        return result
    except Exception as e:
        # Ensure cleanup in case of error
        try:
            p.disconnect()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend files at the end
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
