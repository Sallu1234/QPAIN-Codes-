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

def simulate_surface_code(distance, base_physical_error, num_shots=15000, architecture="rotated"):
    base_physical_error = min(max(base_physical_error, 0.0), 0.5)
    
    task_string = f"surface_code:{architecture}_memory_x"
    
    circuit = stim.Circuit.generated(
        code_task=task_string,
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

def calculate_threshold(results_d3, results_d5, x_axis_rates):
    diff = np.array(results_d3) - np.array(results_d5)
    sign_flips = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_flips) > 0:
        idx = sign_flips[0]
        x1, x2 = x_axis_rates[idx], x_axis_rates[idx+1]
        y1, y2 = diff[idx], diff[idx+1]
        return x1 - y1 * ((x2 - x1) / (y2 - y1))
    return None

print("\n--> Initiating Topology Comparison Benchmarking Sequences...")

photon_loss_rates = np.linspace(0.005, 0.10, 15)
surface_distances = [3, 5]

concat_rotated = {d: [] for d in surface_distances}
concat_unrotated = {d: [] for d in surface_distances}

for kappa in photon_loss_rates:
    print(f"    Processing Environment Coherence Baseline: Loss Rate kappa = {kappa:.3f}")
    
    p_gkp_residual = evaluate_gkp_residual_error(kappa)
    
    for d in surface_distances:
        log_err_rot = simulate_surface_code(distance=d, base_physical_error=p_gkp_residual, architecture="rotated")
        concat_rotated[d].append(log_err_rot)
        
        log_err_unrot = simulate_surface_code(distance=d, base_physical_error=p_gkp_residual, architecture="unrotated")
        concat_unrotated[d].append(log_err_unrot)

thresh_rotated = calculate_threshold(concat_rotated[3], concat_rotated[5], photon_loss_rates)
thresh_unrotated = calculate_threshold(concat_unrotated[3], concat_unrotated[5], photon_loss_rates)

plt.figure(figsize=(11, 7.5))

colors = {3: 'tab:blue', 5: 'tab:red'}

for d in surface_distances:
    plt.plot(photon_loss_rates, concat_unrotated[d], '--', marker='s', color=colors[d],
             label=f"Unrotated Lattice (d={d})", linewidth=2.0, alpha=0.8)

for d in surface_distances:
    plt.plot(photon_loss_rates, concat_rotated[d], '-', marker='o', color=colors[d],
             label=f"Rotated Lattice (d={d})", linewidth=2.5)

if thresh_unrotated:
    plt.axvline(x=thresh_unrotated, color='gray', linestyle='--', linewidth=1.5)
    plt.text(thresh_unrotated - 0.002, 5e-5, f"Unrotated threshold approx {thresh_unrotated:.3f}", 
             rotation=90, color='gray', fontweight='bold')

if thresh_rotated:
    plt.axvline(x=thresh_rotated, color='black', linestyle='-.', linewidth=1.5)
    plt.text(thresh_rotated + 0.001, 5e-5, f"Rotated threshold approx {thresh_rotated:.3f}", 
             rotation=90, color='black', fontweight='bold')

plt.yscale('log')
plt.xlim([0.0, 0.10])  
plt.ylim([1e-5, 1.0])

plt.xlabel("Physical Environmental Photon Loss Rate (kappa)", fontsize=13)
plt.ylabel("Macro-Architecture Logical Error Rate (Lambda)", fontsize=13)
plt.title("Topological Architecture Comparison:\nRotated vs. Unrotated Concatenated GKP-Surface Codes", fontsize=14)

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11, loc="lower right")
plt.tight_layout()

plt.savefig('concatenated_topology_comparison.pdf', dpi=300)
plt.show()

print("\n--> Topology comparison execution successful. Vector file exported.")