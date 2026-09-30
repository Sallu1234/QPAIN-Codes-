import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.0625
ampl_cutoff = 1e-3
total_steps = 15
cutoff = 20

def simulate_gkp(error_rate, ec_interval):
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    
    init_prog = sf.Program(3)
    with init_prog.context as modes:
        GKP([0, 0], epsilon, ampl_cutoff=ampl_cutoff) | modes[0]
    result = eng.run(init_prog)
    initial_state = result.state

    rho0 = initial_state.reduced_dm(0)
    eigvals, eigvecs = np.linalg.eigh(rho0)
    psi0 = eigvecs[:, np.argmax(eigvals)]

    fidelities = [1.0]
    state = initial_state

    for t in range(1, total_steps + 1):
        step_prog = sf.Program(3)
        with step_prog.context as modes:
            dq = np.random.normal(0, error_rate)
            dp = np.random.normal(0, error_rate)
            Xgate(dq) | modes[0]
            Zgate(dp) | modes[0]

            do_ec = (t % ec_interval == 0)
            if do_ec:
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
        state = result.state

        rho_t = state.reduced_dm(0)
        eigvals, eigvecs = np.linalg.eigh(rho_t)
        psi_t = eigvecs[:, np.argmax(eigvals)]
        fidelity = np.abs(np.vdot(psi0, psi_t))**2
        fidelities.append(fidelity)

    return fidelities

error_rates = [0.05, 0.1, 0.2, 0.3, 0.5]
fixed_ec_interval = 2

plt.figure(figsize=(10, 5))
for er in error_rates:
    fid = simulate_gkp(er, fixed_ec_interval)
    plt.plot(range(len(fid)), fid, '-o', label=f"error_rate={er}")
plt.xlabel("Step", fontsize=14)
plt.ylabel("Fidelity", fontsize=14)
plt.title(f"Fidelity vs Steps for Different Noise Strengths (EC interval={fixed_ec_interval})", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

ec_intervals = [1, 2, 4, 5, 6]
fixed_error_rate = 0.1

plt.figure(figsize=(10, 5))
for ec in ec_intervals:
    fid = simulate_gkp(fixed_error_rate, ec)
    plt.plot(range(len(fid)), fid, '-o', label=f"EC interval={ec}")
plt.xlabel("Step", fontsize=14)
plt.ylabel("Fidelity", fontsize=14)
plt.title(f"Fidelity vs Steps for Different EC Intervals (error_rate={fixed_error_rate})", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()