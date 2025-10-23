from enum import Enum, auto


class Phase(Enum):
    PRETRAIN = auto()  # task-learning on modulations, ssl on weights
    TASK_LEARNING = auto()  # task-learning on modulations
    CONSOLIDATION = auto()  # ssl + TMCL on weights
