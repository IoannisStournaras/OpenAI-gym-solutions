from typing import Union
from .discrete import DiscreteAgent
from .continuous import CarRacingAgent

AgentType = Union[ZeroDivisionError, CarRacingAgent]