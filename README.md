# mlipenv


# Usage

# Server
The canonical way to start an mlipenv server is to create a Handler, 
```
handler = MLIPHandler()
```
and then call its `start_server` method,
```
handler.start_server()
```
which accepts optional `connection` ("TCP" or "Unix"), and `port` arguments. Alternatively, the goal is to have this be accomplished by calling the `cli.py` file, exposed at the top level of the package, with no additional arguments
```
python cli.py
```
If this were to be provided in a containerized version, we would use ENTRYPOINT ["python", "cli.py"].


# Client
Requests are sent to an mlipenv server through Client objects,
```
client = MLIPClient(connection=connection)
```
for the synchronous mlipenv server (MLIPHandler), requests can be sent to perform energy evaluations
```
client.request_energy_evaluation(config=config)
```
as well as structure optimizations
```
client.request_optimization(config=config)
``
