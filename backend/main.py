"""
Backend API. Three jobs:
1. Let the webcam simulator push fresh counts       -> POST /occupancy/{train}/update
2. Let the passenger webapp read the current status -> GET  /occupancy/{train}
3. Let a booking action be checked against crowd level, and rejected
   if the train is already Overcrowded                -> POST /booking/{train}

Run: uvicorn backend.main:app --reload --port 8000
Docs auto-generated at: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import occupancy_store

app = FastAPI(title="Train Crowd Density API")

# Allow the passenger webapp (served separately) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a prototype; restrict this in a real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


class CountUpdate(BaseModel):
    count: int


@app.get("/occupancy/{train_number}")
def get_occupancy(train_number: str):
    return occupancy_store.get_status(train_number)


@app.post("/occupancy/{train_number}/update")
def update_occupancy(train_number: str, update: CountUpdate):
    """Called by the webcam/entry-counter simulator, not by passengers."""
    occupancy_store.set_count(train_number, update.count)
    return occupancy_store.get_status(train_number)


@app.post("/booking/{train_number}")
def book_ticket(train_number: str):
    """
    The ticket-limiting logic: reject new bookings once a train is
    Overcrowded. This is intentionally a hard rule for the prototype -
    a real system might instead throttle bookings gradually as the
    High threshold approaches, rather than a single on/off switch.
    """
    status = occupancy_store.get_status(train_number)

    if status["level"] == "Overcrowded":
        raise HTTPException(
            status_code=409,
            detail=f"Booking rejected: train {train_number} is Overcrowded "
                    f"({status['count']}/{status['capacity']}). Please choose another train.",
        )

    occupancy_store.increment_count(train_number)
    new_status = occupancy_store.get_status(train_number)
    return {"message": "Booking confirmed", "status": new_status}
