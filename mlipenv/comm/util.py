
def infer_mode(connection):
    import mlipenv.comm.enums as enums
    if isinstance(connection, tuple):
        addr, port = connection
        if (
            isinstance(addr, str) 
            and (isinstance(port, int) 
                 or (isinstance(port, str) and port.strip().isdecimal()))
        ):
            mode = enums.ServerModes.TCP
    elif isinstance(connection, str):
        mode = enums.ServerModes.Unix
    else:
        raise ValueError(f"invalid connection spec {connection}")
    return mode


def resolve_connection(address=None, port=None, connection=None):
    import ast
    if connection is None:
        if port is not None:
            connection = (address, port)
        else:
            connection = address
    elif isinstance(connection, str):
        connection = ast.literal_eval(connection)
    return connection