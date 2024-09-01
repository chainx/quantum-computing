from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


import json
import numpy as np
import matplotlib.pyplot as plt

API_TOKEN = json.load(open('/media/chainx/seagate/Programming/API tokens/qiskit.json'))["API_TOKEN"]
SERVICE = QiskitRuntimeService(channel="ibm_quantum", token=API_TOKEN)
BACKEND = SERVICE.backend(name="ibm_brisbane")

def main():
    max_qubits = 41

    # qc_list = []
    # for n in range(7, max_qubits+1, 2):
    #     qc_list.append(create_circuit(num_qubits=n))

    # qc_transpiled_list = optimize(qc_list)
    # job_id = execute_on_hardware(qc_transpiled_list)
    # print(job_id)

    post_process_results("cva63pe8gpc0008x634g", max_qubits)

# ==================================================================================

def create_circuit(num_qubits):
    num_ancilla = num_qubits-2
    num_ancilla_pairs = num_ancilla//2

    qr = QuantumRegister(num_qubits)
    cr1 = ClassicalRegister(num_ancilla_pairs+1, name="cr1") # Parity controlled X-gate
    cr2 = ClassicalRegister(num_ancilla - num_ancilla_pairs, name="cr2") # Parity controlled Z-gate
    cr3 = ClassicalRegister(2, name="cr3") # For final measurement of control and target qubits
    qc = QuantumCircuit(qr, cr1, cr2, cr3)

    # Initialise control qubit
    qc.h(0)
    qc.barrier()

    # Entangle control qubit and first ancilla qubit
    qc.cx(0, 1)

    for n in range(1, num_qubits//2):
        qc.h(2*n)
        qc.cx(2*n, 2*n+1)
    qc.barrier()
    for n in range(1, num_qubits//2 + 1):
        qc.cx(2*n-1, 2*n)
        qc.h(2*n-1)
    qc.barrier()

    # Measure odd and even ancilla qubits
    for n in range(num_ancilla_pairs+1):
        qc.measure(2*n+1, cr1[n])
        if n==0:
            parity_control = expr.lift(cr1[0])
        else:
            parity_control = expr.bit_xor(cr1[n], parity_control)
    for n in range(num_ancilla_pairs):
        qc.measure(2*n+2, cr2[n])
        if n==0:
            parity_target = expr.lift(cr2[0])
        else:
            parity_target = expr.bit_xor(cr2[n], parity_target)

    # Apply conditional gates
    with qc.if_test(parity_control):
        qc.z(0)
    with qc.if_test(parity_target):
        qc.x(-1)

    # Final measurements on the control and target qubits
    qc.measure(0, cr3[0])
    qc.measure(-1, cr3[1])

    return qc

def optimize(qc_list):
    return generate_preset_pass_manager(backend=BACKEND, optimization_level=1).run(qc_list)

def execute_on_hardware(qc_transpiled_list):
    sampler = SamplerV2(mode=BACKEND)
    job = sampler.run(qc_transpiled_list)
    return job.job_id()

def post_process_results(job_id, max_qubits):
    job = SERVICE.job(job_id)
    prob_Bell, prob_not_Bell = [], []
    for n in range((max_qubits-7)//2+1):
        data = job.result()[n].data.cr3
        counts = data.get_counts()
        prob_Bell.append( (counts['00']+counts['11'])/data.num_shots )
        prob_not_Bell.append( (counts['01']+counts['10'])/data.num_shots )

    plt.plot(range(7, max_qubits+1, 2), prob_Bell, '--o', label="00 or 11")
    plt.plot(range(7, max_qubits+1, 2), prob_not_Bell, '-.^', label="01 or 10")
    plt.xlabel("Number of qubits")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

# ==================================================================================

if __name__=="__main__":
    main()