
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import traceback
import logging

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