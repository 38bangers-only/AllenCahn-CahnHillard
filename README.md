# 2D Phase Field PINNs: Allen-Cahn and Cahn-Hillard 

A PyTorch implementation of Physics-Informed Neural Networks (PINNs) to solve two 2D phase-field equations, that
rains neural networks to solve the **Allen-Cahn** and **Cahn-Hilliard** PDEs without any traditional numerical solver — just autograd and a composite loss function.

- **Allen-Cahn** — models interface motion (circular droplet IC)
- **Cahn-Hilliard** — models spinodal decomposition (conserves mass). A fourth order-PDE, which had to be split into two PDEs with separate independent parameters spatio-temporally.

Both run on a 2D unit square domain with Neumann (zero-flux) boundary conditions.

## How to run
```bash
pip install torch numpy matplotlib
python ACHPINN.py
```

NOTE : To obtain the best results out of the model, preferably run it on a GPU. The script is tailored to detect CUDA

Saves model weights (`ac_model.pth`, `ch_model.pth`) and a results figure (`phase_field_pinns.png`).

## Tunable parameters
```python
EPOCHS = 3000   # more epochs = better results
T      = 0.3    # time horizon
EPS    = 0.05   # interface width
```

## Stack

Python · PyTorch · Matplotlib
