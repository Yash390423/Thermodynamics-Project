# ================================
# DSMC for N2 + CSV DATABASE (multi-pressure)
# ================================

import numpy as np
import math
import pandas as pd
from tqdm import tqdm

# ---------------- CONSTANTS ----------------
kB = 1.380649e-23
Na = 6.02214076e23
R  = 8.314462618

# N2 PROPERTIES
M_molar = 28.0134e-3
m = M_molar / Na                 # kg per molecule
d = 3.64e-10                     # molecular diameter (m)
sigma = math.pi * d**2           # collision cross-section (m^2)

# ---------------- SIMULATION PARAMETERS ----------------
T0 = 300.0                       # initial temperature (K)
V = 1e-6                         # box volume (m^3)
L = V ** (1/3)                   # cubic box side (m)
A_wall = L**2

Nsim = 2000                      # number of simulated particles
dt = 2e-5                        # time step
steps = 2000                     # number of steps
sample_interval = 10

ncx = ncy = ncz = 6              # grid cells for collisions
collision_prefactor = 0.02       # reduces collision attempts

# ---------------- PRESSURE RANGES (user requested) ----------------
# 50 points from 1e-6 to 1e-5 (logspace)
p_range1 = np.logspace(-6, -3, 250)
# 50 points from 1e-2 to 1 (logspace)
p_range2 = np.logspace(-3, 0, 250)

plist = np.concatenate([p_range1, p_range2])
# Create readable case labels
pressures = {f"p{idx:03d}_{p:.1e}Pa": float(p) for idx, p in enumerate(plist)}

# ---------------- OUTPUT ----------------
output_csv = "dsmc_n2_database.csv"

# ---------------- THEORETICAL FUNCTIONS ----------------
def n_theory(p, T):
    return p/(kB*T)

def vbar_theory(T):
    return math.sqrt(8*kB*T/(math.pi*m))

def collision_freq_theory(n, T):
    return n*sigma*vbar_theory(T)

def mfp_theory(n):
    return 1/(math.sqrt(2)*n*sigma)

# ---------------- CELL HELPER ----------------
def get_cells(pos):
    ix = np.clip((pos[:,0]/L * ncx).astype(int), 0, ncx-1)
    iy = np.clip((pos[:,1]/L * ncy).astype(int), 0, ncy-1)
    iz = np.clip((pos[:,2]/L * ncz).astype(int), 0, ncz-1)
    return ix, iy, iz

# ======================================================
#                    MAIN LOOP
# ======================================================
rows = []
np.random.seed(0)

# Outer progress bar over many pressures
for case, p_target in tqdm(pressures.items(), desc="Pressure cases", unit="case"):

    # Print smaller progress for console clarity (tqdm outer already)
    n_real = n_theory(p_target, T0)
    N_real = n_real * V
    weight = N_real / Nsim     # number of real molecules / 1 sim particle

    # Initialize positions and velocities for this case
    pos = np.random.rand(Nsim,3) * L
    v_std = math.sqrt(kB*T0/m)
    vel = np.random.normal(scale=v_std, size=(Nsim,3))

    momentum_x = 0.0
    total_collisions = 0

    for step in range(steps):

        # -------- FREE FLIGHT --------
        pos += vel * dt

        # ---- WALL COLLISIONS (X direction for pressure) ----
        # x = 0
        hit = pos[:,0] < 0
        if hit.any():
            # momentum change for reflected particles (real molecules)
            momentum_x += np.sum(2*m*(-vel[hit,0])) * weight
            vel[hit,0] *= -1
            pos[hit,0] *= -1

        # x = L
        hit = pos[:,0] > L
        if hit.any():
            momentum_x += np.sum(2*m*(vel[hit,0])) * weight
            vel[hit,0] *= -1
            pos[hit,0] = 2*L - pos[hit,0]

        # Y/Z walls (no momentum count for pressure measurement)
        for dim in [1,2]:
            left = pos[:,dim] < 0
            right = pos[:,dim] > L
            if left.any():
                vel[left,dim] *= -1
                pos[left,dim] *= -1
            if right.any():
                vel[right,dim] *= -1
                pos[right,dim] = 2*L - pos[right,dim]

        # -------- COLLISIONS --------
        ix, iy, iz = get_cells(pos)
        cellmap = {}
        for p_idx, key in enumerate(zip(ix,iy,iz)):
            cellmap.setdefault(key, []).append(p_idx)

        for key, members in cellmap.items():
            if len(members) < 2:
                continue

            attempts = max(1, int(collision_prefactor * len(members)**2))

            for _ in range(attempts):
                i, j = np.random.choice(members, 2, replace=False)
                vr = vel[i] - vel[j]
                vr_mag = np.linalg.norm(vr)
                if vr_mag == 0:
                    continue

                # probability of collision
                p_col = sigma * vr_mag * n_real * dt
                if np.random.rand() < p_col:
                    # isotropic scattering
                    u = np.random.normal(size=3)
                    u /= np.linalg.norm(u)
                    vr_new = vr_mag * u
                    vcm = 0.5*(vel[i] + vel[j])
                    vel[i] = vcm + 0.5*vr_new
                    vel[j] = vcm - 0.5*vr_new
                    total_collisions += 1

        # -------- SAMPLE --------
        if step % sample_interval == 0:

            t = (step+1)*dt
            speeds2 = np.sum(vel**2, axis=1)
            mean_speed = np.mean(np.sqrt(speeds2))
            T_sim = m*np.mean(speeds2)/(3*kB)

            P_sim = momentum_x/(A_wall*t) if t>0 else 0.0

            # collision frequency measurement
            if t > 0:
                # total_collisions is number of DSMC collision events;
                # convert to collisions per real molecule per second
                nu_meas = (total_collisions / t) / (Nsim/weight) if (Nsim/weight) > 0 else np.nan
                mfp_meas = mean_speed/nu_meas if nu_meas>0 else np.nan
            else:
                nu_meas = np.nan
                mfp_meas = np.nan

            # theoretical values (based on T0 and n_real)
            n_th = n_real
            v_th = vbar_theory(T0)
            nu_th = collision_freq_theory(n_th, T0)
            mfp_th = mfp_theory(n_th)

            rows.append({
                "case": case,
                "pressure_Pa": p_target,
                "step": step,
                "time_s": t,
                "T_sim_K": T_sim,
                "P_sim_Pa": P_sim,
                "mean_speed_sim": mean_speed,
                "collisions_total": total_collisions,
                "nu_meas": nu_meas,
                "mfp_meas": mfp_meas,
                "n_theory": n_th,
                "nu_theory": nu_th,
                "mfp_theory": mfp_th,
                "mean_speed_theory": v_th,
                "N_real": N_real,
                "weight": weight
            })

# -------- SAVE CSV --------
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print("\nSaved DSMC database:", output_csv)
