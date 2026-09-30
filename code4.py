import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt
import scipy.special as sp

print(f"--> Successfully initialized Stim version: {stim.__version__}")

def evaluate_gkp_residual_error(kappa, squeezing_db=11.5):
    delta_sq = 10**(-squeezing_db / 10.0)
    total_variance = delta_sq + kappa
    distance = np.sqrt(np.pi) / 2.0
    p_err = 0.5 * sp.erfc(distance / np.sqrt(2 * total_variance))
    return max(1e-6, p_err)

def simulate_surface_code(topology, distance, base_physical_error, num_shots=15000):
    base_physical_error = min(max(base_physical_error, 0.0), 0.5)
    code_task = f"surface_code:{topology}"
    
    circuit = stim.Circuit.generated(
        code_task=code_task,
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=base_physical_error,
        after_reset_flip_probability=base_physical_error,
        before_measure_flip_probability=base_physical_error,
        before_round_data_depolarization=base_physical_error
    )
    
    sampler = circuit.compile_detector_sampler()
    syndrome_batch, observable_batch = sampler.sample(shots=num_shots, separate_observables=True)
    
    error_model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(error_model)
    predictions = matching.decode_batch(syndrome_batch)
    
    num_failures = np.sum(observable_batch.flatten() != predictions.flatten())
    return max(num_failures / num_shots, 1e-6)

print("\n--> Initiating Master Architecture Benchmarking Sequences...")

photon_loss_rates = np.linspace(0.01, 0.08, 12)
surface_distances = [3, 5]

results_baseline_rot = {d: [] for d in surface_distances}
results_concat_rot = {d: [] for d in surface_distances}
results_concat_unrot = {d: [] for d in surface_distances}

for kappa in photon_loss_rates:
    print(f"    Processing Environmental Loss: kappa = {kappa:.3f}")
    
    p_gkp = evaluate_gkp_residual_error(kappa)
    scaled_raw_error = kappa * 0.1  
    
    for d in surface_distances:
        err_base = simulate_surface_code("rotated_memory_x", d, scaled_raw_error)
        results_baseline_rot[d].append(err_base)
        
        err_rot = simulate_surface_code("rotated_memory_x", d, p_gkp)
        results_concat_rot[d].append(err_rot)
        
        err_unrot = simulate_surface_code("unrotated_memory_x", d, p_gkp)
        results_concat_unrot[d].append(err_unrot)

plt.figure(figsize=(12, 8))

colors_base = {3: 'darkgray', 5: 'black'}
for d in surface_distances:
    plt.plot(photon_loss_rates, results_baseline_rot[d], ':', marker='s', color=colors_base[d],
             label=f"Baseline Qubit + Rotated Surface (d={d})", alpha=0.6, linewidth=2)

colors_rot = {3: 'tab:blue', 5: 'tab:red'}
for d in surface_distances:
    plt.plot(photon_loss_rates, results_concat_rot[d], '-', marker='o', color=colors_rot[d],
             label=f"Concatenated GKP+Steane + Rotated (d={d})", linewidth=2.5)

colors_unrot = {3: '#aec7e8', 5: '#ff9896'}
for d in surface_distances:
    plt.plot(photon_loss_rates, results_concat_unrot[d], '--', marker='^', color=colors_unrot[d],
             label=f"Concatenated GKP+Steane + Unrotated (d={d})", linewidth=2.5)

plt.yscale('log')
plt.xlim([0.01, 0.08])  
plt.ylim([1e-5, 1.0])

plt.xlabel("Physical Environmental Photon Loss Rate (kappa)", fontsize=13)
plt.ylabel("Macro-Architecture Logical Error Rate (Lambda)", fontsize=13)
plt.title("Macro-Architecture Comparison:\nBaseline vs. Concatenated GKP+Steane on Rotated and Unrotated Lattices", fontsize=14)

plt.grid(True, which="both", ls="--", alpha=0.5)

plt.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.savefig('master_architecture_comparison.pdf', dpi=300)
plt.show()

print("\n--> Master architecture execution successful. Vector file exported.")