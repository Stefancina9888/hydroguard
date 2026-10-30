"""Validate JSON-native raw samples before any NumPy conversion can erase types."""

import math
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def finite_number(value):
    # bool is an int subclass: isinstance(value, (int, float)) is insufficient.
    if type(value) not in (int, float):
        raise ValueError("Each sample must be a real JSON number; booleans are forbidden")
    try:
        if not math.isfinite(value):
            raise ValueError("Each sample must be finite")
        return float(value)
    except OverflowError as error:
        raise ValueError("Each sample must be representable as a finite float64") from error


def sample_list(value):
    if type(value) is not list:
        raise ValueError("Each sensor must be a one-dimensional JSON array of numbers")
    return value


Sample = Annotated[float, BeforeValidator(finite_number)]
At100Hz = Annotated[
    list[Sample], Field(min_length=6000, max_length=6000), BeforeValidator(sample_list)
]
At10Hz = Annotated[
    list[Sample], Field(min_length=600, max_length=600), BeforeValidator(sample_list)
]
At1Hz = Annotated[list[Sample], Field(min_length=60, max_length=60), BeforeValidator(sample_list)]


class Sensors(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    PS1: At100Hz
    PS2: At100Hz
    PS3: At100Hz
    PS4: At100Hz
    PS5: At100Hz
    PS6: At100Hz
    EPS1: At100Hz
    FS1: At10Hz
    FS2: At10Hz
    TS1: At1Hz
    TS2: At1Hz
    TS3: At1Hz
    TS4: At1Hz
    VS1: At1Hz


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sensors: Sensors
