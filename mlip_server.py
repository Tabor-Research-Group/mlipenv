import logging
from mlipenv.client import MLIPHandler

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import sys, os, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port",
                        default=os.environ.get(MLIPHandler.DEFAULT_PORT_ENV_VAR, os.environ.get("SESSION_ID")))
    parser.add_argument("-c", "--no_server", action="store_true", default=False)
    parser.add_argument("--allow_scipts",
                        action="store_true",
                        default=False)
    parser.add_argument("request_args",
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.no_server:
        import subprocess
        subprocess.run(["conda", "run", "--no-capture-output", "-n", "fairchem"] + args.request_args)

    port = args.port
    if port is None:
        raise ValueError(f"`{MLIPHandler.DEFAULT_PORT_ENV_VAR}` must be set at the environment level")
    port = MLIPHandler.get_valid_port(port)

    MLIP_CONNECTION = ('localhost', MLIPHandler.get_valid_port(port))
    
    try: # implicit server startup ping done for every request
        MLIPHandler.start_server(connection=MLIP_CONNECTION)
    except OSError: # server exists
        if not len(args.request_args):
            print(f"Already serving on {MLIP_CONNECTION}")
        pass
    if len(args.request_args):
        MLIPHandler.client_request(args.request_args[0], args.request_args[1:], connection=MLIP_CONNECTION)