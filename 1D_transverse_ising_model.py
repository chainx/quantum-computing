from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit.circuit.library import YGate, UnitaryGate

import json
import numpy as np
import matplotlib.pyplot as plt

API_TOKEN = json.load(open('/media/chainx/seagate/Programming/API tokens/qiskit.json'))["API_TOKEN"]
SERVICE = QiskitRuntimeService(channel="ibm_quantum", token=API_TOKEN)
BACKEND = SERVICE.backend(name="ibm_brisbane")

# THE CIRCUITS IN THIS SCRIPT ARE BASED ON THE 1D VERSION OF THE TRANSVERSE ISING MODEL
# CONSIDERED IN THE IBM PAPER ON QUANTUM UTILITY: https://www.nature.com/articles/s41586-023-06096-3

sqrt_Y = UnitaryGate(YGate().power(1/2), label=r'$\sqrt{Y}$')
sqrt_Y_dg = UnitaryGate(sqrt_Y.inverse(), label=r'$\sqrt{Y}^\dag$')

num_qubits = 100
num_trotter_steps = 10
measured_qubits = [49, 50]
rx_angle = np.pi/2

def main():
    # job_id = submit_run()
    # print(job_id)
    job_id = "cv91xn68gpc0008ggga0" # "cv90134sgfsg008eb75g" # "cv91bddsgfsg008ebewg"
    post_processing(job_id)

def submit_run():
    # STEP 1: Map the problem to circuits and observables
    qc_list = generate_1d_tfim_circuit(layer_barriers=True)
    
    # STEP 2: Optimize
    qc_transpiled_list = transpile(qc_list, backend=BACKEND, optimization_level=1)

    # STEP 3: Execute on hardware
    sampler = SamplerV2(mode=BACKEND)
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    job = sampler.run(qc_transpiled_list)
    
    return job.job_id()

def post_processing(job_id):
    # STEP 4: Post-processing and plotting
    job = SERVICE.job(job_id)

    survival_probability_list = []
    for trotter_step in range(num_trotter_steps):
        try:
            data = job.result()[trotter_step].data.c
            survival_probability_list.append(data.get_counts()['0'*len(measured_qubits)] / data.num_shots)
        except:
            survival_probability_list.append(0)

    plt.plot(list(range(0, 4*num_trotter_steps, 4)), survival_probability_list, '--o')
    plt.xlabel('2Q gate depth')
    plt.ylabel('Survival probability of the all-0 bitstring')
    plt.xticks(np.arange(0, 44, 4))
    plt.show()

# ==================================================================================

def generate_1d_tfim_circuit(layer_barriers=False):
    qc = QuantumCircuit(num_qubits, len(measured_qubits))
    qc_list = []
    for trotter_step in range(num_trotter_steps):
            add_1d_tfim_trotter_layer(qc, rx_angle, layer_barriers)
            add_1d_tfim_mirrored_trotter_layer(qc, rx_angle, layer_barriers)
            qc.measure(measured_qubits, list(range(len(measured_qubits))))
            qc_list.append(qc)
    return qc_list

def add_1d_tfim_trotter_layer(qc, rx_angle, layer_barriers):
    # Adding R_ZZ gates
    for odd_or_even in [0, 1]:
        for i in range(odd_or_even, qc.num_qubits-1, 2):
            qc.sdg([i, i+1]) # sdg is S^\dag
            qc.append(sqrt_Y, [i+1])
            qc.cx(i, i+1)
            qc.append(sqrt_Y_dg, [i+1])
        if layer_barriers:
            qc.barrier()
    # Adding R_X gates
    qc.rx(rx_angle, list(range(qc.num_qubits)))
    if layer_barriers:
        qc.barrier()

def add_1d_tfim_mirrored_trotter_layer(qc, rx_angle, layer_barriers):
    # Adding R_X gates
    qc.rx(-rx_angle, list(range(qc.num_qubits)))
    if layer_barriers:
        qc.barrier()
    # Adding R_ZZ gates
    for odd_or_even in [1, 0]:
        for i in range(odd_or_even, qc.num_qubits-1, 2):
            qc.append(sqrt_Y, [i+1])
            qc.cx(i, i+1)
            qc.append(sqrt_Y_dg, [i+1])
            qc.s([i, i+1])
        if layer_barriers:
            qc.barrier()

# ==================================================================================

if __name__=="__main__":
    main()