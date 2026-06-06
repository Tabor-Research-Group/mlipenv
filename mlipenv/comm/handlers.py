
import os
import json
import abc
import threading
import subprocess
import socketserver
import traceback

from mlipenv.comm.servers import NodeCommUnixServer, NodeCommTCPServer
from mlipenv.comm.util import infer_mode
from mlipenv.comm.clients import BaseClient
from mlipenv.comm.queuing.job_palette import JobScheduler
import mlipenv.comm.enums as enums
from mlipenv.comm.queuing.util import register_as_async

__all__ = [
    "NodeCommHandler",
    "ShellCommHandler",
    "MLIPHandler",
    "AsyncMLIPHandler",
]

class NodeCommHandler(socketserver.StreamRequestHandler):

    def handle(self):
        try:
            # self.rfile is a file-like object created by the handler;
            # we can now use e.g. readline() instead of raw recv() calls
            self.data = self.rfile.readline().strip()
            response = self.handle_json_request(self.data)
            # Likewise, self.wfile is a file-like object used to write back
            # to the client
        except:
            response = {
                "stdout": "",
                "stderr": traceback.format_exc(limit=10)
            }
        try:
            self.wfile.write(json.dumps(response).encode() + b'\n')
        except:
            traceback.print_exc(limit=10)  # big ol' fallback

    def handle_json_request(self, message: bytes):
        try:
            request = json.loads(message.decode())
        except:
            response = {
                "stdout": "",
                "stderr": traceback.format_exc(limit=10)
            }
        else:
            comm = request.get("command", '<unknown>')
            args = request.get("args", [])
            env = request.get("env", {})
            print(f"Got: {comm} {args}")
            response = self.dispatch_request(request, env)
            print(f"Sending: {response}")

        return response

    @property
    def method_dispatch(self): 
        return dict(
            {
                enums.Methods.CD.value: self.change_pwd,
                enums.Methods.PWD.value: self.get_pwd,
                enums.Methods.EXIT.value: self.stop_server,
                enums.Methods.SHUTDOWN.value: self.stop_server
            },
            **self.subclass_methods()
        )
    def change_pwd(self, args):
        os.chdir(args[0])
        return {
            'stdout':"",
            'stderr':""
        }
    def get_pwd(self, args):
        cwd = os.getcwd()
        return {
            'stdout':cwd,
            'stderr':""
        }
    def setup_env(self, env):
        if 'pwd' in env:
            os.chdir(env['pwd'])
    def get_methods(self) -> 'dict[str,method]':
        return self.method_dispatch
    def dispatch_request(self, request: dict, env:dict):
        method = request.get("command", None)
        if method is None:
            response = {
                "stdout": "",
                "stderr": f"no command specified"
            }
        else:
            caller = self.method_dispatch.get(method.lower(), None)
            if caller is None:
                response = {
                    "stdout": "",
                    "stderr": f"unknown command {method}"
                }
            else:
                args = request.get("args", None)
                if args is None:
                    response = {
                        "stdout": "",
                        "stderr": f"malformatted request {request}"
                    }
                else:
                    try:
                        self.setup_env(env)
                        response = caller(*args)
                    except:
                        response = {
                            "stdout": "",
                            "stderr": traceback.format_exc(limit=10)
                        }

        return response

    @classmethod
    def subprocess_response(cls, command, args):
        pipes = subprocess.Popen([command, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        std_out, std_err = pipes.communicate()
        return {
            "stdout":std_out.strip().decode(),
            "stderr":std_err.strip().decode()
        }
    @abc.abstractmethod
    def subclass_methods(self) -> 'dict[str,method]':
        ...

    @staticmethod
    def get_valid_port(port, min_port=4000, max_port=65535):
        port = int(port)
        if port > max_port:
            port = port % max_port
        if port < min_port:
            port = max_port - (port % (max_port - min_port))
        return port

    DEFAULT_CONNECTION = ("localhost", 9999)
    DEFAULT_PORT_ENV_VAR = None
    DEFAULT_SOCKET_ENV_VAR = None
    @classmethod
    def infer_connection(cls, connection, port):
        if connection is None and cls.DEFAULT_SOCKET_ENV_VAR is not None:
            connection = os.environ.get(cls.DEFAULT_SOCKET_ENV_VAR)
        if connection is None:
            if port is None and cls.DEFAULT_PORT_ENV_VAR:
                port = os.environ.get(cls.DEFAULT_PORT_ENV_VAR)
            if port is not None:
                connection = ('localhost', cls.get_valid_port(port))
        if connection is None:
            connection = cls.DEFAULT_CONNECTION
        return connection

    @classmethod
    def set_server(cls, server):
        cls.server = server
    
    @classmethod
    def get_server(cls):
        return cls.server

    TCP_SERVER = NodeCommTCPServer
    UNIX_SERVER = NodeCommUnixServer
    @classmethod
    def get_server_type(cls, mode):
        if mode == "TCP":
            server_type = cls.TCP_SERVER
        elif mode == "Unix":
            server_type = cls.UNIX_SERVER
        else:
            raise NotImplementedError(mode)
        return server_type

    @classmethod
    def start_server(cls, connection=None, port=None, scheduler=False):
        # start the server; default binding is to localhost on port 9999
        connection = cls.infer_connection(connection, port)
        mode = infer_mode(connection)
        print(f"Starting server at {connection} over {mode}")
        server_type = cls.get_server_type(mode)
        scheduler = JobScheduler() if scheduler else None
        with server_type(connection, cls, scheduler) as server:
            cls.set_server(server)
            # Activate the server; this will keep running until you
            # interrupt the program with Ctrl-C
            server.serve_forever()
            if mode == "Unix":
                try:
                    os.remove(connection)
                except OSError:
                    ...

    @classmethod
    def _shutdown_server(cls):
        server = cls.get_server()
        server.shutdown()
        cls.set_server(None)

    @classmethod
    def stop_server(cls, args):
        threading.Thread(target=cls._shutdown_server).start()
        return {
            "stdout": "shutting down server...",
            "stderr": ""
        }

    client_class = BaseClient
    @classmethod
    def client_request(cls, *args, client_class=None, connection=None):
        if client_class is None:
            client_class = cls.client_class
        if connection is None:
            connection = cls.DEFAULT_CONNECTION
        return client_class(connection).communicate(*args)

class ShellCommHandler(NodeCommHandler):

    @abc.abstractmethod
    def get_subprocess_call_list(self):
        ...

    def subclass_methods(self) -> 'dict[str,method]':
        return {
            k:self._wrap_subprocess_call(v)
            for k,v in self.get_subprocess_call_list()
        }

    def _wrap_subprocess_call(self, command):
        if isinstance(command, str):
            def command(*args, _cmd=command, **kwargs):
                return self.subprocess_response(_cmd, *args, **kwargs)
        elif not callable(command):
            def command(*args, _cmd=command, **kwargs):
                return self.subprocess_response(*_cmd, *args, **kwargs)
        return command


class MLIPHandler(NodeCommHandler):

    DEFAULT_PORT_ENV_VAR = 'MLIP_SOCKET_PORT'

    def subclass_methods(self) -> 'dict[str,method]':
        return {
            enums.MLIPMethods.EVALUATE.value: self.evaluate,
        }

    def evaluate(self, config=None, method=None, *args):
        if config is None:
            response = {
                "stdout": "",
                "stderr": "no args provided"
            }
        else:
            try:
                from mlipenv.exec.manager import execute_mlip_job
                response = execute_mlip_job(config, method=method)
                response = {
                    "stdout": "" if response is None else response,
                    "stderr": ""
                } 
            except Exception:
                response = {
                    "stdout": "",
                    "stderr": traceback.format_exc(limit=10)
                }
        return response
    
    @classmethod
    def start_server(cls, connection=None, port=None):
        super().start_server(connection=connection, port=port, scheduler=False)

class AsyncMLIPHandler(NodeCommHandler):

    DEFAULT_PORT_ENV_VAR = 'MLIP_SOCKET_PORT'

    def subclass_methods(self) -> 'dict[str,method]':
        return {
            enums.AsyncMLIPMethods.EVALUATE.value: self.evaluate,
            enums.AsyncMLIPMethods.STATUS.value: self.check_job_status,
            enums.AsyncMLIPMethods.CHECK_JOB_STATUS.value: self.check_job_status,
            enums.AsyncMLIPMethods.CANCEL.value: self.cancel_job,
            enums.AsyncMLIPMethods.CANCEL_JOB.value: self.cancel_job,
        }

    @register_as_async
    def evaluate(self, config=None, method=None, *args):
        if config is None:
            response = {
                "stdout": "",
                "stderr": "no args provided"
            }
        else:
            try:
                # brittle reference
                job_id = self.server.scheduler.submit_job("mlipenv.exec.manager", config)
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
    
    @classmethod
    def start_server(cls, connection=None, port=None):
        super().start_server(connection=connection, port=port, scheduler=True)