from typing import Optional
from dataclasses import dataclass

from litgpt.args import TrainArgs


@dataclass
class FineTuningArgs(TrainArgs):
    learning_rate: Optional[float] = 2e-5
