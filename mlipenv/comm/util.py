
def infer_mode(connection):
    if isinstance(connection, tuple):
        addr, port = connection
        if (
            isinstance(addr, str) 
            and (isinstance(port, int) 
                 or (isinstance(port, str) and port.strip().isdecimal()))
        ):
            mode = "TCP"
    elif isinstance(connection, str):
        mode = "Unix"
    else:
        raise ValueError(f"invalid connection spec {connection}")
    return mode


def resolve_connection(connection, address, port):
    import ast
    if connection is None:
        if port is not None:
            connection = (address, port)
        else:
            connection = address
    elif isinstance(connection, str):
        connection = ast.literal_eval(connection)
    return connection