import socketserver
import uuid
import os
import subprocess
import threading
import logging
import traceback
from enum import Enum
import json
import abc

import yaml

logger = logging.getLogger(__name__)

class statuses(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"

class JobUnit:
    def __init__(self, job_id: str, func: str, args: str):
        self.job_id = job_id
        self.func = func
        self.args = self.unpack_args(args)
        self.status = statuses.PENDING
        self.backend = args.get("environment", {}).get("backend", {})
        self.slurm_job_id = None
        self.pid = None
        self.pgid = None
        self.returncode = None
        self.stdout = ""
        self.stderr = ""
        self.error = ""

    def get_status(self) -> str:
        return self.status
    def set_status(self, status: str):
        self.status = status
    
    def unpack_args(self, args: str) -> str:
        if args.endswith(".yaml") or args.endswith(".yml"):
            try:
                with open(args, "r") as f:
                    args = yaml.safe_load(f)
            except Exception:
                raise
        elif args.endswith(".json"):
            try:
                with open(args, "r") as f:
                    args = json.load(f)
            except Exception:
                raise
        else:
            raise NotImplementedError
        return args

    def to_dict(self):
        return {
        "job_id": self.job_id,
        "func": self.func,
        "args": self.args,
        "status": self.status.value,
        "backend": self.backend,
        "slurm_job_id": self.slurm_job_id,
        "pid": self.pid,
        "pgid": self.pgid,
        "returncode": self.returncode,
        "stdout": self.stdout,
        "stderr": self.stderr,
        "error": self.error,
        }


class JobExecutor(abc.ABC):
    @abc.abstract_method
    def _construct_cmd(self):
        ...
    @abc.abstract_method
    def submit_job(self, job):
        ...
    @abc.abstract_method
    def monitor_job(self, job_id):
        ...
    @abc.abstract_method
    def cancel_job(self, job_id):
        ...

class JobScheduler(socketserver.ThreadingTCPServer):
    def __init__(self):
        self.jobs = {}

    def _construct_cmd(self, job):
        func = job.func
        args = job.args
        env_args = args.get("environment", {})
        env = os.environ.copy()
        for k,v in env_args.get("vars", {}).items():
            env[str(k).upper()] = str(v)
        cmd = []
        if "conda" in env_args:
            env_name = env_args["conda"]
            cmd = ["conda", "run", "-n", env_name]
        elif "venv" in env_args:
            env_name = env_args["venv"]
            cmd = [env_name]
        elif "container" in env_args:
            env_name = env_args["container"]
            cmd = ["singularity", "run", env_name]
        else:
            logger.warning("Environment as either conda, venv path, or container path has not been specified.")
        
        runtime_args = args.get("runtime", {})
        cmd = cmd + ["python", "-m", func, json.dumps(runtime_args)]
        
        sbatch_directives = " ".join(f"--{k}={v}" for k, v in env_args.get("sbatch", {}).items())
        if sbatch_directives:
            cmd = ["sbatch", "--export=ALL", sbatch_directives, f'--wrap="{" ".join(s for s in cmd)}"']
        return cmd, env
    
    def _monitor_slurm_job(self, job_id):
        ...
    
    def _monitor_local_job(self, job_id, process):
        try:
            stdout, stderr = process.communicate()
            
            job = self.jobs.get(job_id)
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
            logger.exception(f"Failed while monitoring job with job_id: {job_id}.")
            job = self.jobs.get(job_id)
            if job is not None:
                job.set_status(statuses.FAILED)
                job.error = traceback.format_exc(limit=10)

    def submit_job(self, func, args):
        job_id = uuid.uuid4().hex
        job = JobUnit(job_id, func, args)
        self.jobs[job_id] = job

        cmd, env = self._construct_cmd(job)
        logger.info(f"Starting subprocess command {cmd}")
        process = subprocess.Popen(
            cmd, 
            env=env, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        job.pid = process.pid
        job.set_status(statuses.RUNNING)
        t = threading.Thread(
            target=self._monitor_job, 
            args=(job_id, process),
            daemon=True
        )
        t.start()

        return job_id
        
    def query_job(self, job_id=None):
        return {
            this_job_id: job.to_dict() 
            for this_job_id, job in self.jobs.items() 
            if job_id is None or this_job_id == job_id
        }
        