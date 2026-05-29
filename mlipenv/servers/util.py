
def register_as_async(func):
    func.is_asynchronous = True
    return func