def register_as_async(func):
    func.is_asynchronous = True
    return func


def unpack_args(args: str) -> dict:
    import json, yaml
    if args.endswith(".yaml") or args.endswith(".yml"):
        try:
            with open(args, "r") as f:
                args = yaml.safe_load(f)
        except Exception:
            raise
    elif args.endswith(".json"):
        try:
            with open(args, "r") as f:
                args = json.load(f)
        except Exception:
            raise
    else:
        raise NotImplementedError
    return args

