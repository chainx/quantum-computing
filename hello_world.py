from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorOptions, EstimatorV2
from qiskit_aer import Aer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import json
import matplotlib.pyplot as plt

API_TOKEN = json.load(open('/media/chainx/seagate/Programming/API tokens/qiskit.json'))["API_TOKEN"]
SERVICE = QiskitRuntimeService(channel="ibm_quantum", token=API_TOKEN)
N=100

JOB_IDS = {
    "ibm_kyoto": "cv8s3f5emvv00085005g",
    "ibm_brisbane": "cv8s9wqkfn8g008vq0f0",
}

def main():
    # results = submit_quantum_computation(simulation=True)
    # plot_results(results)
    
    plot_results(job_id=JOB_IDS["ibm_brisbane"])

# ==================================================================================

def plot_results(results=None, job_id=None):
    if job_id:
        job = SERVICE.job(job_id)
        results = job.result()[0].data.evs

    values = [res / results[0] for res in results]

    plt.scatter(range(1, N), values, marker='o', label=f'{N}-qubit GHZ state')
    plt.title("Testing spin coherence over distance")
    plt.xlabel("Distance between qubits 0 and $i$")
    plt.ylabel(r'$\langle Z_0 Z_i \rangle \ / \ \langle Z_0 Z_1 \rangle$')
    plt.show()

def submit_quantum_computation(simulation=True, QPU_name="ibm_brisbane"):

    qc = QuantumCircuit(N) # N Qubit circuit
    qc.h(0)                # Apply Hadamard gate to the 0 qubit
    for n in range(N-1):
        qc.cx(n, n+1)      # Apply CNOT to 0th and nth qubits

    if simulation:
        backend = Aer.get_backend('aer_simulator')
    else:
        backend = SERVICE.backend(name=QPU_name)

    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
    qc_transpiled = pass_manager.run(qc)

    observable_labels = ["Z" + "I"*i + "Z" + "I"*(N-2-i) for i in range(N-1)]
    observables = [SparsePauliOp(observable) for observable in observable_labels]
    observables_transpiled = [ob.apply_layout(qc_transpiled.layout) for ob in observables]

    if simulation:
        options = None
    else:
        options = EstimatorOptions()
        options.resilience_level = 1
        options.optimization_level = 0 # Because transpilation already done locally
        options.dynamical_decoupling.enable = True # Inserts pulses into idle time
        options.dynamical_decoupling.sequence_type = "XY4"

    estimator = EstimatorV2(backend, options=options)
    job = estimator.run([(qc_transpiled, observables_transpiled)])

    if simulation:
        return job.result()[0].data.evs
    else:
        return job.job_id()

if __name__=="__main__":
    main()