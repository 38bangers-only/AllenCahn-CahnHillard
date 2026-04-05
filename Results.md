<img width="2867" height="1896" alt="phase_field_pinns" src="https://github.com/user-attachments/assets/b0c8b3c9-21f0-4259-ae1e-56c4747325a1" />
## Results

> Trained for 3000 epochs · ε = 0.05 · T = 0.3 · domain [0,1]²

### Allen-Cahn
The network correctly captures a circular droplet (φ ≈ +1 inside, φ ≈ −1 outside)
with a diffuse interface. The droplet shrinks gradually over time as expected from
the physics. Loss converges well — IC loss reaches ~1e-5 and BC loss ~1e-4 by epoch 3000,
though the PDE residual plateaus around ~1e-3, indicating room for improvement.

### Cahn-Hilliard
Phase separation from a noisy initial condition is captured qualitatively, with φ and μ
fields developing structured domains over time. However, the chemical potential field
remains somewhat noisy, and the physics diagnostics panel shows the conserved mass
(CH mass) is not perfectly flat — it drifts slightly — meaning the network hasn't fully
satisfied the conservation law yet.

### Physics Diagnostics
- **AC bulk energy** rises over time rather than decaying monotonically, pointing to
  underfitting in the PDE loss.
- **CH mass** shows a drift instead of staying constant, indicating the Cahn-Hilliard
  model needs more training to enforce conservation.

---

## Getting better results

The current run is a quick proof-of-concept. The following changes are expected to
improve accuracy significantly:

| Parameter | Current | Suggested |
|---|---|---|
| `EPOCHS` | 3000 | 8000 – 15000 |
| Collocation points (`n` in `_loss_pde`) | 2000 – 3000 | 5000 – 8000 |
| `w_ic` (IC loss weight) | 10.0 | 20.0 – 50.0 |
| `w_mu` (CH chemical potential weight) | 5.0 | 10.0 – 20.0 |
| Network width | 128 | 256 |

**Other things worth trying:**
- Switch to **L-BFGS** after Adam converges — PINNs often benefit from a second-stage
  L-BFGS fine-tuning pass for tighter PDE residuals
- **Increase Fourier features** (`n_freq`) from 32 to 64 for better high-frequency
  resolution in the Cahn-Hilliard field
- **Adaptive collocation** — sample more points near the interface where gradients
  are steep, rather than uniform random sampling
