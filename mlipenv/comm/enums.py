from enum import Enum

class Methods(Enum):
    CD = "cd"
    PWD = "pwd"
    EXIT = "exit"
    SHUTDOWN = "shutdown"

class MLIPMethods(Enum):
    EVALUATE = "evaluate"

class AsyncMLIPMethods(Enum):
    EVALUATE = "evaluate"
    STATUS = "status"
    CHECK_JOB_STATUS = "check_job_status"
    CANCEL = "cancel"
    CANCEL_JOB = "cancel_job"

class MLIPEvaluate(Enum):
    OPTIMIZATION = "optimization"
    ENERGY_EVALUATION = "energy"