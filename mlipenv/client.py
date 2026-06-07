
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import traceback
import logging
import subprocess

from .servers.node_comm import *

class MLIPHandler(NodeCommHandler):

    DEFAULT_PORT_ENV_VAR = 'MLIP_SOCKET_PORT'

    def get_methods(self) -> 'dict[str,method]':
        return {
            "evaluate": self.evaluate,
        }

    @contextmanager
    def redirect_logging(self, buffer, logger=None, enabled=True):
        if enabled:
            root = logging.getLogger(logger)
            root.setLevel(logging.INFO)
            handler = logging.StreamHandler(buffer)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            try:
                root.addHandler(handler)
                yield None
            finally:
                root.removeHandler(handler)
                handler.close()

    CAPTURE_LOGS = True
    TRACEBACK_LIMIT = None
    def evaluate(self, args):
        from mlipenv.mlip_opt import call_to_mlip_server

        if not len(args):
            response = {
                "stdout": "",
                "stderr": "no args provided"
            }
        else:
            buffer = StringIO()
            try:
                with self.redirect_logging(buffer, enabled=self.CAPTURE_LOGS), redirect_stdout(buffer), redirect_stderr(buffer):
                    res = call_to_mlip_server(*args)
                response = {
                    "stdout": buffer.getvalue(),
                    "stderr": ""
                }
                if res is not None:
                    response = response | res
            except:
                response = {
                    "stdout": buffer.getvalue(),
                    "stderr": traceback.format_exc(limit=self.TRACEBACK_LIMIT)
                }

        return response

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import sys, os, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port",
                        default=os.environ.get(MLIPHandler.DEFAULT_PORT_ENV_VAR, os.environ.get("SESSION_ID")))
    parser.add_argument("-c", "--no_server", action="store_true", default=False)
    parser.add_argument("-e", "--env", default=None)
    # parser.add_argument("--allow_scipts",
    #                     action="store_true",
    #                     default=False)
    parser.add_argument("request_args",
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.no_server:
        env = args.env
        if args.env is None:
            env = os.environ.get("MLIP_DEFAULT_ENVIRONMENT")
        if isinstance(args.env, str) and args.env.lower() == "none":
            env = None

        if env is None:
            subprocess.run(args.request_args)
        else:
            subprocess.run(["conda", "run", "--no-capture-output", "-n", args.env] + args.request_args)
    else:
        port = args.port
        if port is None:
            raise ValueError(f"`{MLIPHandler.DEFAULT_PORT_ENV_VAR}` must be set at the environment level")
        port = MLIPHandler.get_valid_port(port)

        MLIP_CONNECTION = ('localhost', MLIPHandler.get_valid_port(port))

        try:  # implicit server startup ping done for every request
            MLIPHandler.start_server(connection=MLIP_CONNECTION)
        except OSError:  # server exists
            if not len(args.request_args):
                print(f"Already serving on {MLIP_CONNECTION}")
            pass
        if len(args.request_args):
            MLIPHandler.client_request(args.request_args[0], args.request_args[1:],
                                       connection=MLIP_CONNECTION,
                                       print_response=True)

if __name__ == "__main__":
    main()