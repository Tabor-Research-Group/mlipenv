import socket
import socketserver

class NodeCommTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, connection, handler_cls, scheduler=None):
        super().__init__(connection, handler_cls)
        self.scheduler = scheduler

class NodeCommUnixServer(socketserver.ThreadingUnixStreamServer):
    allow_reuse_address = True

    def __init__(self, connection, handler_cls, scheduler=None):
        super().__init__(connection, handler_cls)
        self.scheduler = scheduler

    def server_bind(self):
        """Called by constructor to bind the socket.
        May be overridden.
        """

        if self.allow_reuse_address:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()
    
    