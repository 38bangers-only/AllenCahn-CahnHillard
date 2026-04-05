# 2D Phase Field PINNs: Allen-Cahn and Cahn-Hillard 
# Phase-Field PINNs: Allen-Cahn & Cahn-Hilliard

A PyTorch implementation of Physics-Informed Neural Networks (PINNs) to solve two 2D phase-field equations, that
rains neural networks to solve the **Allen-Cahn** and **Cahn-Hilliard** PDEs without any traditional numerical solver — just autograd and a composite loss function.

- **Allen-Cahn** — models interface motion (circular droplet IC)
- **Cahn-Hilliard** — models spinodal decomposition (conserves mass)

Both run on a 2D unit square domain with Neumann (zero-flux) boundary conditions. This was a quite interesting and challenging project as it required modelling two fourth-order PDEs, which were split into two second-order non parabolic PDEs for breaking down complexity.

## How to run
```bash
pip install torch numpy matplotlib
python ACHPINN.py
```

Saves model weights (`ac_model.pth`, `ch_model.pth`) and a results figure (`phase_field_pinns.png`).

## Tunable parameters
```python
EPOCHS = 3000   # more epochs = better results
T      = 0.3    # time horizon
EPS    = 0.05   # interface width
```

## Stack

Python · PyTorch · Matplotlib
