import traceback

from mlipenv.servers.node_comm import *
from mlipenv.servers.util import register_as_async

class MLIPHandler(NodeCommHandler):

    DEFAULT_PORT_ENV_VAR = 'MLIP_SOCKET_PORT'

    def subclass_methods(self) -> 'dict[str,method]':
        return {
            "evaluate": self.evaluate,
            "status": self.check_job_status,
            "check_job_status": self.check_job_status,
            "cancel": self.cancel_job,
            "cancel_job": self.cancel_job,
        }

    @register_as_async
    def evaluate(self, config=None, *args):
        if config is None:
            response = {
                "stdout": "",
                "stderr": "no args provided"
            }
        else:
            # print("yo! :3")
            try:
                # brittle reference
                job_id = self.server.scheduler.submit_job("mlipenv.manager", config)
                response = {
                    "stdout": f"job has been submitted. job_id = {job_id}",
                    "stderr": ""
                } 
            except Exception:
                response = {
                    "stdout": "",
                    "stderr": traceback.format_exc(limit=10)
                }
        return response 
    
    def check_job_status(self, job_id=None, *args):
        try:
            job_info = self.server.scheduler.query_job(job_id)

            response = {
                "stdout": f"Status(es):\n{'\n'.join(f"{job_id}: {job['status']}" for job_id, job in job_info.items())}",
                "stderr": ""
            } 
        except Exception:
            response = {
                "stdout": "",
                "stderr": traceback.format_exc(limit=10)
            }
        return response
    
    def cancel_job(self, job_id, *args):
        try:
            response = self.server.scheduler.cancel_job(job_id)
            response = {
                "stdout": response,
                "stderr": ""
            }
        except Exception:
            response = {
                "stdout": "",
                "stderr": traceback.format_exc(limit=10)
            }
        return response


if __name__ == "__main__":
    import sys, os, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port",
                        default=os.environ.get(MLIPHandler.DEFAULT_PORT_ENV_VAR, os.environ.get("SESSION_ID")))
    parser.add_argument("--no_server", action="store_true", default=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("request_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.no_server:
        import subprocess
        runtime_args = args.config if args.config is not None else args.request_args
        subprocess.run(["conda", "run", "--no-capture-output", "-n", "fairchem"] + runtime_args)

    port = args.port
    if port is None:
        raise ValueError(f"`{MLIPHandler.DEFAULT_PORT_ENV_VAR}` must be set at the environment level")
    port = MLIPHandler.get_valid_port(port)

    MLIP_CONNECTION = ('localhost', MLIPHandler.get_valid_port(port))
    
    if not len(args.request_args):
        try: # implicit server startup ping done for every request
            MLIPHandler.start_server(connection=MLIP_CONNECTION)
        except OSError: # server exists
            if not len(args.request_args):
                print(f"Already serving on {MLIP_CONNECTION}")
            pass
    else:
        try:
            method, method_args = args.request_args[0], args.request_args[1:]
            method_args = method_args if args.config is None else args.config
            MLIPHandler.client_request(method, method_args, connection=MLIP_CONNECTION)
        except Exception:
            raise