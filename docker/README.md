## Install Docker Engine
There are several way availables  for  installing the Docker Engine. Look at this page https://docs.docker.com/engine/install/

## Run the Qibo container
Execute the command

```
$ docker run -it --rm javierserrano/qibo:bash
```
## Play with the Qibo examples
Once you are running the container you can play with the examples

```
root@1a57b52b754a:/usr/src/app# cd examples/grover3sat/

root@1a57b52b754a:/usr/src/app/examples/grover3sat# python main.py

Qubits encoding the solution: 10

Total number of qubits used:  19

Most common bitstring: 0110000101

Exact cover solution:  0110000101
```
