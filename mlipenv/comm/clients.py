import sys
import os
import json
import socket

from mlipenv.comm.util import infer_mode, resolve_connection
import mlipenv.comm.enums as enums

class BaseClient:
    def __init__(
            self, 
            address: str = None,
            port: int = None,
            connection: str | tuple = None,
            timeout: float = 10, 
    ):
        self.conn = resolve_connection(address=address, port=port, connection=connection)
        mode = infer_mode(self.conn)
        if mode == enums.ServerModes.TCP:
            self.mode = socket.AF_INET
        elif mode == enums.ServerModes.Unix:
            self.mode = socket.AF_UNIX
        else:
            raise NotImplementedError(mode)
        self.timeout = timeout

    SEND_CWD = True
    def prep_command_env(self):
        env = {}
        if self.SEND_CWD:
            env['pwd'] = os.getcwd()
        return env

    def communicate(self, command, args):
        request = json.dumps({
            "command": command,
            "args": args,
            "env": self.prep_command_env()
        }) + "\n"
        request = request.encode()

        # Create a socket (SOCK_STREAM means a TCP socket)
        mode = infer_mode(self.conn)
        # print(f"Sending request over {mode}")
        if mode == "Unix" and not os.path.exists(self.conn):
            raise ValueError(f"socket file {self.conn} doesn't exist")
        with socket.socket(self.mode, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect(self.conn)
            sock.settimeout(self.timeout)
            sock.sendall(request)
            # Receive data from the server and shut down
            # if stuff is being printed to the console, we could send it intermittently?
            sock.settimeout(100000)
            body = b''
            while b'\n' not in body:
                body = body + sock.recv(1024)

        response = json.loads(body.strip().decode())
        msg = response.get("stdout","")
        if len(msg) > 0: print(msg, file=sys.stdout)
        msg = response.get("stderr","")
        if len(msg) > 0: print(msg, file=sys.stderr)

class MLIPClient(BaseClient):
    def __init__(
            self,
            use_server: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_server = use_server

    # def prep_config_with_method(self, config, method_type):
    #     return {
    #         "method": method_type,
    #         "config": config,
    #     }

    def request_optimization(self, config):
        # config = self.prep_config_with_method(config, enums.MLIPEvaluate.OPTIMIZATION.value)
        self.communicate(enums.MLIPMethods.EVALUATE.value, (config, enums.MLIPEvaluate.OPTIMIZATION.value))

    def request_energy_evaluation(self, config):
        # config = self.prep_config_with_method(config, enums.MLIPEvaluate.ENERGY_EVALUATION.value)
        self.communicate(enums.MLIPMethods.EVALUATE.value, (config, enums.MLIPEvaluate.ENERGY_EVALUATION.value))
