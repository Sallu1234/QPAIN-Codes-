import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.0625
ampl_cutoff = 1e-3
total_steps = 20
ec_interval = 2
error_rate = 0.1
cutoff = 20 

eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})

init_prog = sf.Program(3)
with init_prog.context as modes:
    GKP([0, 0], epsilon, ampl_cutoff=ampl_cutoff) | modes[0]

init_result = eng.run(init_prog)
initial_state = init_result.state
print("Initial state: GKP Logical |0⟩")
print(f"GKP parameters: epsilon={epsilon}, ampl_cutoff={ampl_cutoff}")

rho0 = initial_state.reduced_dm(0)
eigvals, eigvecs = np.linalg.eigh(rho0)
psi0 = eigvecs[:, np.argmax(eigvals)] 

x = np.linspace(-5, 5, 100)
p = np.linspace(-5, 5, 100)
sqrt_pi = np.sqrt(np.pi)

wigner0 = initial_state.wigner(0, x, p)
plt.figure(figsize=(6, 5))
contour = plt.contourf(x, p, wigner0, 50, cmap="RdBu_r", levels=50)
plt.title("Initial GKP |0⟩ State", fontsize=14)
plt.xlabel("Position (q)", fontsize=12)
plt.ylabel("Momentum (p)", fontsize=12)
plt.colorbar(contour)
for n in range(-3, 4):
    for m in range(-3, 4):
        plt.plot(n*sqrt_pi, m*sqrt_pi, 'k+', markersize=5, alpha=0.5)
plt.grid(True, alpha=0.2)
plt.show()

initial_fidelity = 1.0
print(f"Initial self-fidelity: {initial_fidelity:.4f}")

state_history = [("Initial", initial_state, initial_fidelity)]

for t in range(1, total_steps + 1):
    step_prog = sf.Program(3)
    with step_prog.context as modes:
        dq = np.random.normal(0, error_rate)
        dp = np.random.normal(0, error_rate)
        Xgate(dq) | modes[0]
        Zgate(dp) | modes[0]
        print(f"Step {t}: Applying errors dq={dq:.3f}, dp={dp:.3f}")

        do_ec = (t % ec_interval == 0)
        if do_ec:
            print(f"Step {t}: Applying Steane EC")
            GKP([0, 0], epsilon, ampl_cutoff=ampl_cutoff) | modes[1]
            GKP([0, 0], epsilon, ampl_cutoff=ampl_cutoff) | modes[2]
            Rgate(np.pi) | modes[2] 

            CZgate() | (modes[0], modes[1])
            Rgate(-np.pi/2) | modes[0]
            CZgate() | (modes[0], modes[2])
            Rgate(+np.pi/2) | modes[0]

            MeasureHomodyne(np.pi/2, select=0.0) | modes[1]
            MeasureHomodyne(np.pi/2, select=0.0) | modes[2]

    result = eng.run(step_prog)
    current_state = result.state

    rho_t = current_state.reduced_dm(0)
    eigvals, eigvecs = np.linalg.eigh(rho_t)
    psi_t = eigvecs[:, np.argmax(eigvals)]
    fidelity = np.abs(np.vdot(psi0, psi_t))**2
    print(f"Step {t}: Fidelity = {fidelity:.4f}")

    state_history.append((f"Step {t}{' (EC)' if do_ec else ''}", current_state, fidelity))

    if do_ec:
        wigner = current_state.wigner(0, x, p)
        plt.figure(figsize=(6, 5))
        contour = plt.contourf(x, p, wigner, 50, cmap="RdBu_r", levels=50)
        plt.title(f"Step {t} - After EC\nFidelity = {fidelity:.4f}", fontsize=14)
        plt.xlabel("q", fontsize=12)
        plt.ylabel("p", fontsize=12)
        plt.colorbar(contour)
        for n in range(-3, 4):
            for m in range(-3, 4):
                plt.plot(n*sqrt_pi, m*sqrt_pi, 'k+', markersize=5, alpha=0.5)
        plt.grid(True, alpha=0.2)
        plt.show()

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

states_to_plot = [0] 
ec_indices = [i for i, (name, _, _) in enumerate(state_history) if 'EC' in name]
states_to_plot += ec_indices[:5]

for idx, plot_idx in enumerate(states_to_plot):
    name, state, fid = state_history[plot_idx]
    wigner = state.wigner(0, x, p)
    axes[idx].contourf(x, p, wigner, 30, cmap="RdBu_r")
    axes[idx].set_title(f"{name}\nFidelity={fid:.4f}", fontsize=12)
    axes[idx].set_xlabel("q", fontsize=10)
    axes[idx].set_ylabel("p", fontsize=10)
    for n in range(-3, 4):
        for m in range(-3, 4):
            axes[idx].plot(n*sqrt_pi, m*sqrt_pi, 'k+', markersize=3, alpha=0.3)
    axes[idx].grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

print("\nGKP Logical |0⟩ characteristics:")
print("1. Grid pattern in phase space with peaks at q,p = n√π")
print("2. Negative Wigner values between peaks (non-classical)")
print("3. Protected against small displacements by error correction")
print("4. Ideal peaks would be delta functions; finite ε gives finite width")

steps = list(range(len(state_history)))
fidelities = [fid for _, _, fid in state_history]

plt.figure(figsize=(10, 5))
plt.plot(steps, fidelities, 'b-o', linewidth=2, markersize=6)
plt.axhline(y=1.0, color='r', linestyle='--', label='Initial Fidelity')
plt.xlabel('Step', fontsize=14)
plt.ylabel('Fidelity', fontsize=14)
plt.title('Fidelity Evolution vs Time Steps', fontsize=16)
plt.xticks(range(len(state_history)), [name for name, _, _ in state_history], rotation=45, fontsize=10)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()