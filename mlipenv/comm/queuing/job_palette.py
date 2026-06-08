import socketserver
import uuid
import os
import subprocess
import time
import threading
import logging
import traceback
import signal
from enum import Enum
import json
import abc

from mlipenv.comm.queuing.util import unpack_args

logger = logging.getLogger(__name__)

class statuses(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"
    MISSING = "missing"

JOB_UNIT_REGISTRY = {}

def register_job_unit(runtime_manager: str, unit_factory=None):
    if unit_factory is None:
        def register(unit_factory):
            return register_job_unit(runtime_manager, unit_factory)
        return register
    JOB_UNIT_REGISTRY[runtime_manager] = unit_factory
    return unit_factory

def get_job_unit(runtime_manager: str):
    return JOB_UNIT_REGISTRY[runtime_manager]

@register_job_unit("base")
class JobUnit:
    def __init__(self, job_id: str, func: str, args: dict):
        self.job_id = job_id
        self.func = func
        self.args = args
        self.backend = None
        self.runtime_manager = None
        self.status = statuses.PENDING
        self.stdout = ""
        self.stderr = ""
        self.error = ""

    def get_status(self) -> str:
        return self.status
    def set_status(self, status: str):
        self.status = status

    def to_dict(self) -> dict:
        return {
        "job_id": self.job_id,
        "func": self.func,
        "args": self.args,
        "backend": self.backend,
        "runtime_manager": self.runtime_manager,
        "status": self.status.value,
        "stdout": self.stdout,
        "stderr": self.stderr,
        "error": self.error,
        }
    
    def is_incomplete(self) -> bool:
        return self.status == statuses.PENDING or self.status == statuses.RUNNING

@register_job_unit("local")
class LocalJobUnit(JobUnit):
    def __init__(self, job_id: str, func: str, args: dict):
        super().__init__(job_id, func, args)
        self.runtime_manager = "local"
        self.executor = LocalJobExecutor()
        self.pid = None
        self.returncode = None
    
    def to_dict(self) -> bool:
        return super().to_dict() | {
            "pid": self.pid,
            "returncode": self.returncode,
        }

@register_job_unit("slurm")
class SlurmJobUnit(JobUnit):
    def __init__(self, job_id: str, func: str, args: dict):
        super().__init__(job_id, func, args)
        self.runtime_manager = "slurm"
        self.executor = SlurmJobExecutor()
        self.slurm_job_id = None
        self.slurm_job_status = None
        self.slurm_seff_information = None

    def to_dict(self):
        return super().to_dict() | {
            "slurm_job_id": self.slurm_job_id,
            "slurm_job_status": self.slurm_job_status,
            "slurm_seff_information": self.slurm_seff_information,
        }
        

BASE_CMD_REGISTRY = {
    "conda": ["conda", "run", "-n", None],
    "venv": [None],
    "container": ["singularity", "run", None],
}

def get_base_cmd(package_bearer_dict: dict) -> list:
    if package_bearer_dict:
        package_bearer_type, package_bearer_loc = next(iter(package_bearer_dict.items()))
        if package_bearer_type in BASE_CMD_REGISTRY:
            cmd = BASE_CMD_REGISTRY[package_bearer_type]
            return [package_bearer_loc if el is None else el for el in cmd]
    logger.warning("Environment as either conda, venv path, or container path was either not found or not specified.")
    return []

class JobExecutor(abc.ABC):
    def _construct_cmd(self, job: JobUnit) -> tuple[list, dict]:
        func = job.func
        args = job.args
        env_args = args.get("environment", {})
        env = os.environ.copy()
        for k,v in env_args.get("vars", {}).items():
            env[str(k).upper()] = str(v)
        package_bearer_dict = env_args.get("package_bearer", {})
        cmd = get_base_cmd(package_bearer_dict)
        
        job_config_args = args.get("job", {}).get("config", {})
        cmd = cmd + ["python", "-m", func, json.dumps(job_config_args)]
        return cmd, env
    
    def get_monitor(self) -> threading.Thread | None:
        return self.monitor if hasattr(self, "monitor") else None
            
    def submit_job(self, job: JobUnit):
        cmd, env = self._construct_cmd(job)
        logger.info(f"Starting subprocess command {cmd}")
        process = subprocess.Popen(
            cmd, 
            env=env, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True
        )

        job.pid = process.pid
        job.set_status(statuses.RUNNING)
        monitor = threading.Thread(
            target=self.monitor_job, 
            args=(job, process),
            daemon=True
        )
        monitor.start()
        self.monitor = monitor
    
    @abc.abstractmethod
    def monitor_job(self, job: JobUnit, process: subprocess.Popen):
        ...
    
    @abc.abstractmethod
    def cancel_job(self, job: JobUnit, timeout: int | float):
        ...

class LocalJobExecutor(JobExecutor):
    KILL_PROCESS_POLL_TIME=0.1

    def __init__(self, kill_process_poll_time: int | float | None = None):
        self.kill_process_poll_time = self.KILL_PROCESS_POLL_TIME if kill_process_poll_time is not None else kill_process_poll_time

    def monitor_job(self, job: LocalJobUnit, process: subprocess.Popen):
        try:
            stdout, stderr = process.communicate()
            job = job.backend.get_job(job.job_id)
            if job is None:
                return

            job.returncode = process.returncode
            job.stdout = stdout
            job.stderr = stderr
            
            if process.returncode == 0:
                job.set_status(statuses.COMPLETED)
            else:
                job.set_status(statuses.FAILED)

        except Exception:
            logger.exception(f"Failed while monitoring job with job_id: {job.job_id}.")
            job = self.jobs.get(job.job_id)
            if job is not None:
                job.set_status(statuses.FAILED)
                job.error = traceback.format_exc(limit=10)

    def _is_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    def cancel_job(self, job: LocalJobUnit, timeout: int | float):
        if not self._is_process_alive(job.pid):
            return 
        try:
            os.killpg(job.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        
        time_elapsed = 0.0
        while time_elapsed < timeout:
            if not self._is_process_alive(job.pid):
                return
            time.sleep(self.kill_process_poll_time)
            time_elapsed += self.kill_process_poll_time
        
        try:
            os.killpg(job.pid, signal.SIGKILL)
        except ProcessLookupError:
            return

class SlurmJobExecutor(JobExecutor):
    SLURM_JOB_POLL_TIME=10

    def __init__(self, slurm_job_poll_time: int | float | None = None):
        self.slurm_job_poll_time = self.SLURM_JOB_POLL_TIME if slurm_job_poll_time is None else slurm_job_poll_time
    
    def _construct_cmd(self, job: SlurmJobUnit) -> tuple[list,dict]:
        cmd, env = super()._construct_cmd(job)
        sbatch_args = job.args.get("environment", {}).get("sbatch", {})
        sbatch_directives = " ".join(f"--{k}={v}" for k, v in sbatch_args.items())
        if sbatch_directives:
            cmd = ["sbatch", "--export=ALL", sbatch_directives, f'--wrap="{" ".join(s for s in cmd)}"']
        return cmd, env
    
    def get_slurm_job_query_cmd(self, slurm_job_id: str, sacct: bool = False) -> list:
        # return ["squeue", "-j", slurm_job_id, --format?]
        ...
        
    def parse_slurm_job_query_response(self, response: str) -> dict:
        ...

    def is_slurm_incomplete_status(self, job_slurm_status: str) -> bool:
        return job_slurm_status == "PENDING" or job_slurm_status == "RUNNING"
    
    def is_slurm_success_status(self, job_slurm_status: str) -> bool:
        return job_slurm_status == "COMPLETE"
    
    def slurm_job_postprocess(self, job: SlurmJobUnit):
        ...

    def update_job_status(self, job: SlurmJobUnit, slurm_job_status: str):
        if self.is_slurm_success_status(slurm_job_status):
            job.status = statuses.COMPLETED
        else:
            job.status = statuses.FAILED
        job.slurm_job_status = slurm_job_status
    
    def monitor_slurm_job(self, job: SlurmJobUnit):
        slurm_job_query_cmd = self.get_slurm_job_query_cmd(job.slurm_job_id)
        while job.is_incomplete():
            process = subprocess.run(slurm_job_query_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            parsed_slurm_job_query = self.parse_slurm_job_query_response(process.stdout)
            if job.slurm_job_id in parsed_slurm_job_query:
                slurm_job_status = parsed_slurm_job_query[job.slurm_job_id]["status"]
                if self.is_slurm_incomplete_status(slurm_job_status):
                    time.sleep(self.slurm_job_poll_time)
                else:
                    self.update_job_status(job, slurm_job_status)
            else:
                # `squeue` does not report for terminated jobs. hopefully the job is in `sacct`.
                slurm_job_query_cmd = self.get_slurm_job_query_cmd(job.slurm_job_id, sacct=True)
                process = subprocess.run(slurm_job_query_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                parsed_slurm_job_query = self.parse_slurm_job_query_response(process.stdout)
                if job.slurm_job_id in parsed_slurm_job_query:
                    slurm_job_status = parsed_slurm_job_query[job.slurm_job_id]["status"]
                else:
                    # this should not be reachable
                    job.status = statuses.MISSING
                    return
        
        self.slurm_job_postprocess(job)

    def monitor_job(self, job: SlurmJobUnit, process: subprocess.Popen):
        try:
            stdout, stderr = process.communicate()
            
            job = job.backend.get_job(job.job_id)
            if job is None:
                return
            
            slurm_job_id = stdout.split(" ")[-1]
            job.slurm_job_id = slurm_job_id
            self.monitor_slurm_job(job)

            job.returncode = process.returncode
            job.stdout = stdout
            job.stderr = stderr
            
            if process.returncode == 0:
                job.set_status(statuses.COMPLETED)
            else:
                job.set_status(statuses.FAILED)
            
        except Exception:
            logger.exception(f"Failed while monitoring job with job_id: {job.job_id}.")
            job = self.jobs.get(job.job_id)
            if job is not None:
                job.set_status(statuses.FAILED)
                job.error = traceback.format_exc(limit=10)
        
    def cancel_job(self, job: SlurmJobUnit, timeout: int | float):
        ...

class JobBackend(abc.ABC):
    @abc.abstractmethod
    def add_job(self, job: JobUnit):
        ...
    @abc.abstractmethod
    def cancel_job(self, job_id: str):
        ...
    @abc.abstractmethod
    def query_job(self, job_id: str = None) -> dict:
        ...
    @abc.abstractmethod
    def get_job(self, job_id: str = None) -> JobUnit | None:
        ...
    
class DictJobBackend(JobBackend):
    def __init__(self):
        self.jobs = {}
    
    def add_job(self, job: JobUnit, job_id: str):
        self.jobs[job_id] = job

    def cancel_job(self, job_id: str, timeout: int | float):
        if job_id not in self.jobs:
            return
        job = self.jobs[job_id]
        # having to pass ,job, here is ugly
        job.executor.cancel_job(job, timeout)

    def query_job(self, job_id: str = None) -> dict:
        return {
            this_job_id: job.to_dict() 
            for this_job_id, job in self.jobs.items() 
            if job_id is None or this_job_id == job_id
        }

    def get_job(self, job_id: str = None) -> JobUnit | None:
        return self.jobs.get(job_id, None)

class JobScheduler(socketserver.ThreadingTCPServer):
    def __init__(self):
        self.backend = DictJobBackend()

    def cancel_job(self, job_id: str, timeout: int | float = 10):
        self.backend.cancel_job()

    def submit_job(self, func: str, args: str):
        logger.info("handling job submission...")
        job_id = uuid.uuid4().hex
        args = unpack_args(args)
        runtime_manager = args.get("runtime", {}).get("manager", "local")
        job = get_job_unit(runtime_manager=runtime_manager)(job_id, func, args)
        job.backend = self.backend
        self.backend.add_job(job, job.job_id)
        job.executor.submit_job(job)
        return job_id
        
    def query_job(self, job_id: str = None) -> dict:
        return self.backend.query_job()
        