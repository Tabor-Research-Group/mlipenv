
import os
import argparse

from mlipenv.comm.handlers import *


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
