from io import StringIO
from contextlib import redirect_stdout
import traceback

from servers.node_comm import *

class MLIPHandler(NodeCommHandler):

    DEFAULT_PORT_ENV_VAR = 'MLIP_SOCKET_PORT'

    def get_methods(self) -> 'dict[str,method]':
        return {
            "evaluate": self.evaluate,
        }
    
    def evaluate(self, args):
        from src.mlip_opt import call_to_mlip_server
        if not len(args):
            response = {
                "stdout": "",
                "stderr": "no args provided"
            }
        else:
            try:
                buffer = StringIO()
                with redirect_stdout(buffer):
                    call_to_mlip_server(*args)
                response = {
                    "stdout": buffer.getvalue(),
                    "stderr": ""
                }
            except:
                response = {
                    "stdout": buffer.getvalue(),
                    "stderr": traceback.format_exc(limit=10)
                }
        return response 

if __name__ == "__main__":
    import sys, os

    port = os.environ.get(MLIPHandler.DEFAULT_PORT_ENV_VAR, os.environ.get("SESSION_ID"))
    if port is None:
        raise ValueError(f"`{MLIPHandler.DEFAULT_PORT_ENV_VAR}` must be set at the environment level")
    port = MLIPHandler.get_valid_port(port)

    MLIP_CONNECTION = ('localhost', MLIPHandler.get_valid_port(port))
    if len(sys.argv) == 1:
        try:
            MLIPHandler.start_server(connection=MLIP_CONNECTION)
        except OSError: # server exists
            print(f"Already serving on {MLIP_CONNECTION}")
            pass
    else:
        MLIPHandler.client_request(sys.argv[1], sys.argv[2:], connection=MLIP_CONNECTION)