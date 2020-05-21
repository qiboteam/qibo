import time
from functions import random_state, entropy

def main():
    start_time = time.time()
    qubits=28
    M=600000
    state=random_state(qubits,M)
    result=entropy(state,qubits,M)
    print(result)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
