"""
A simple handler for running subprocess calls on
different nodes in SLURM systems
"""
import abc
import os
import socket, socketserver, json, traceback, subprocess, threading
import sys

__all__ = [
    "NodeCommTCPServer",
    "NodeCommUnixServer",
    "NodeCommHandler",
    "NodeCommClient"
]

def infer_mode(connection):
    if (
            isinstance(connection, tuple)
            and isinstance(connection[0], str) and isinstance(connection[1], int)
    ):
        mode = "TCP"
    elif isinstance(connection, str):
        mode = "Unix"
    else:
        raise ValueError(f"invalid connection spec {connection}")
    return mode

class NodeCommTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

class NodeCommUnixServer(socketserver.UnixStreamServer):
    allow_reuse_address = True

    def server_bind(self):
        """Called by constructor to bind the socket.

        May be overridden.

        """
        
        if self.allow_reuse_address:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()

class NodeCommClient:
    def __init__(self, connection, timeout=10):
        self.conn = connection
        mode = infer_mode(connection)
        if mode == "TCP":
            self.mode = socket.AF_INET
        elif mode == "Unix":
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

    def communicate(self, command, args, print_response=None):
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
        try:
            body = body.strip().decode()
            response = json.loads(body)
        except:
            raise ValueError(f"couldn't parse {body} as JSON")
        else:
            if print_response is None:
                print_response = len(response) == 2
            if print_response:
                msg = response.get("stdout","")
                if len(msg) > 0: print(msg, file=sys.stdout)
                msg = response.get("stderr","")
                if len(msg) > 0: print(msg, file=sys.stderr)

        return response

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
                "cd": self.change_pwd,
                "pwd": self.get_pwd,
                "exit": self.stop_server,
                "shutdown": self.stop_server
            },
            **self.get_methods()
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
                        response = caller(args)
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
    def get_methods(self) -> 'dict[str,method]':
        ...

    @staticmethod
    def get_valid_port(git_port, min_port=4000, max_port=65535):
        git_port = int(git_port)
        if git_port > max_port:
            git_port = git_port % max_port
        if git_port < min_port:
            git_port = max_port - (git_port % (max_port - min_port))
        return git_port

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
    def start_server(cls, connection=None, port=None):
        # start the server; default binding is to localhost on port 9999
        connection = cls.infer_connection(connection, port)
        mode = infer_mode(connection)
        print(f"Starting server at {connection} over {mode}")
        server_type = cls.get_server_type(mode)
        with server_type(connection, cls) as server:
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

    client_class = NodeCommClient
    @classmethod
    def client_request(cls, *args, client_class=None, connection=None, print_response=None):
        if client_class is None:
            client_class = cls.client_class
        if connection is None:
            connection = cls.DEFAULT_CONNECTION
        return client_class(connection).communicate(*args, print_response=print_response)

class ShellCommHandler(NodeCommHandler):

    @abc.abstractmethod
    def get_subprocess_call_list(self):
        ...

    def get_methods(self) -> 'dict[str,method]':
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
