from pydantic import BaseModel, Field
from typing import List


class SensorReading(BaseModel):
    """One cycle's worth of raw sensor readings for a single engine."""
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictRequest(BaseModel):
    """Up to 5 most recent cycles for one engine, oldest first."""
    recent_cycles: List[SensorReading] = Field(..., min_length=1, max_length=5)


class PredictResponse(BaseModel):
    predicted_rul: float
    cycles_used: int
