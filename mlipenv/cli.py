
import os
import argparse

from mlipenv.comm.handlers import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default=None)
    parser.add_argument("-x", "--connection", default=None)
    parser.add_argument("--no_server", action="store_true", default=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("--asynchronous", action="store_true", default=False)
    parser.add_argument("request_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.no_server:
        import subprocess
        calculator = os.environ.get("CALCULATOR", "base")
        runtime_args = args.config if args.config is not None else args.request_args
        subprocess.run(["conda", "run", "--no-capture-output", "-n", calculator] + runtime_args)

    port = args.port
    # connection is annoying
    connection = args.connection
    handler = AsyncMLIPHandler if args.asynchronous else MLIPHandler

    if not len(args.request_args):
        try:
            handler.start_server(connection=connection, port=port)
        except OSError:
            if not len(args.request_args):
                print(f"Server already exists.")
            pass
        except Exception:
            raise
    else:
        try:
            method, method_args = args.request_args[0], args.request_args[1:]
            method_args = method_args if args.config is None else args.config
            # this connection here is oddly-formed
            handler.client_request(method, method_args, connection=connection)
        except Exception:
            raise