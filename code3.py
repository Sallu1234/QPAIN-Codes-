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

def evaluate_pure_cv_gkp_repetition(kappa, N, squeezing_db=11.5):
    delta_sq = 10**(-squeezing_db / 10.0)
    variance_raw = delta_sq + kappa
    variance_q = variance_raw / float(N)
    variance_p = variance_raw * float(N)
    distance = np.sqrt(np.pi) / 2.0
    p_err_q = 0.5 * sp.erfc(distance / np.sqrt(2 * variance_q))
    p_err_p = 0.5 * sp.erfc(distance / np.sqrt(2 * variance_p))
    p_total_logical = 1.0 - (1.0 - p_err_q) * (1.0 - p_err_p)
    return max(1e-6, p_total_logical)

def simulate_surface_code(distance, base_physical_error, num_shots=15000):
    base_physical_error = min(max(base_physical_error, 0.0), 0.5)
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
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

print("\n--> Running CV Bosonic vs. DV Topological Comparison Sequences...")

photon_loss_rates = np.linspace(0.002, 0.04, 15)
modes_N = [3, 5]
surface_distances = [3, 5]

results_pure_cv_rep = {n: [] for n in modes_N}
results_surface = {d: [] for d in surface_distances}

for kappa in photon_loss_rates:
    print(f"    Processing Baseline Loss: kappa = {kappa:.3f}")
    
    p_gkp_raw = evaluate_gkp_residual_error(kappa)
    
    for n in modes_N:
        err_cv = evaluate_pure_cv_gkp_repetition(kappa, N=n)
        results_pure_cv_rep[n].append(err_cv)
        
    for d in surface_distances:
        err_dv = simulate_surface_code(distance=d, base_physical_error=p_gkp_raw)
        results_surface[d].append(err_dv)

plt.figure(figsize=(11, 7.5))

colors_cv = {3: 'tab:green', 5: 'darkgreen'}
for n in modes_N:
    plt.plot(photon_loss_rates, results_pure_cv_rep[n], ':', marker='s', color=colors_cv[n],
             label=f"Pure CV GKP Repetition Code (Modes N={n})", linewidth=2.5)

colors_dv = {3: 'tab:blue', 5: 'tab:red'}
for d in surface_distances:
    plt.plot(photon_loss_rates, results_surface[d], '-', marker='o', color=colors_dv[d],
             label=f"Concatenated GKP + 2D Rotated Surface Code (d={d})", linewidth=2.5)

plt.yscale('log')
plt.xlim([0.002, 0.04])
plt.ylim([1e-6, 1.0])

plt.xlabel("Physical Environmental Photon Loss Rate (kappa)", fontsize=13)
plt.ylabel("Logical Logical Error Rate (Lambda)", fontsize=13)
plt.title("Continuous-Variable (CV) vs. Discrete-Variable (DV) Architectural Profiles:\nPure Bosonic GKP Repetition vs. Topological GKP-Surface Code", fontsize=13)

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=10, loc="lower right")
plt.tight_layout()

plt.savefig('cv_bosonic_repetition_comparison.pdf', dpi=300)
plt.show()

print("\n--> Comprehensive cross-domain analysis successful. Vector file exported.")