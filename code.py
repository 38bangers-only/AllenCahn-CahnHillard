import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import os,time

torch.manual_seed(40)
np.random.seed(40)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
"""
2D Phase-Field PINNs: Allen-Cahn & Cahn-Hilliard

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import os, time

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# 1. NETWORK ARCHITECTURE
# ─────────────────────────────────────────────
class PhaseFieldNet(nn.Module):
    """Fully-connected PINN with sinusoidal first layer (Fourier features)."""

    def __init__(self, layers: list[int], use_fourier: bool = True, sigma: float = 1.0):
        super().__init__()
        self.use_fourier = use_fourier
        in_dim = layers[0]

        if use_fourier:
            # Random Fourier feature matrix (fixed)
            n_freq = 32
            B = torch.randn(in_dim, n_freq) * sigma
            self.register_buffer("B", B)
            first_in = 2 * n_freq
        else:
            first_in = in_dim

        dims = [first_in] + layers[1:]
        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

        # Xavier init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_fourier:
            proj = x @ self.B
            x = torch.cat([torch.sin(2 * np.pi * proj),
                            torch.cos(2 * np.pi * proj)], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────
# 2. ALLEN-CAHN PINN
# ─────────────────────────────────────────────
class AllenCahnPINN:
    """
    PDE:  ∂φ/∂t = ε²(∂²φ/∂x² + ∂²φ/∂y²) − (φ³ − φ)
    IC:   φ(x,y,0) = φ₀(x,y)
    BC:   Neumann (zero-flux) on all walls
    """

    def __init__(self, eps: float = 0.05, T: float = 0.5):
        self.eps = eps
        self.T = T
        self.model = PhaseFieldNet(layers=[3, 128, 128, 128, 128, 1]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9995)
        self.history = {"total": [], "pde": [], "ic": [], "bc": []}

    # ── helpers ──────────────────────────────
    @staticmethod
    def _initial_condition(x, y):
        """Circular droplet IC: tanh profile."""
        r = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        return torch.tanh((0.25 - r) / 0.05)

    def _predict(self, x, y, t):
        inp = torch.stack([x, y, t], dim=-1)
        return self.model(inp).squeeze(-1)

    def _pde_residual(self, x, y, t):
        x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
        phi = self._predict(x, y, t)

        phi_t = torch.autograd.grad(phi, t, grad_outputs=torch.ones_like(phi),
                                    create_graph=True)[0]
        phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi),
                                    create_graph=True)[0]
        phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi),
                                    create_graph=True)[0]
        phi_xx = torch.autograd.grad(phi_x, x, grad_outputs=torch.ones_like(phi_x),
                                    create_graph=True)[0]
        phi_yy = torch.autograd.grad(phi_y, y, grad_outputs=torch.ones_like(phi_y),
                                    create_graph=True)[0]

        laplacian = phi_xx + phi_yy
        double_well = phi**3 - phi
        residual = phi_t - self.eps**2 * laplacian + double_well
        return residual

    # ── loss terms ───────────────────────────
    def _loss_pde(self, n=3000):
        x = torch.rand(n, device=device)
        y = torch.rand(n, device=device)
        t = torch.rand(n, device=device) * self.T
        r = self._pde_residual(x, y, t)
        return (r**2).mean()

    def _loss_ic(self, n=1000):
        x = torch.rand(n, device=device)
        y = torch.rand(n, device=device)
        t = torch.zeros(n, device=device)
        phi_pred = self._predict(x, y, t)
        phi_true = self._initial_condition(x, y)
        return ((phi_pred - phi_true)**2).mean()

    def _loss_bc(self, n=500):
        """Neumann BC: ∂φ/∂n = 0 on x=0,1 and y=0,1."""
        loss = torch.tensor(0.0, device=device)
        t = torch.rand(n, device=device) * self.T

        for edge, coord, var in [
            ("x0", 0.0, "x"), ("x1", 1.0, "x"),
            ("y0", 0.0, "y"), ("y1", 1.0, "y"),
        ]:
            other = torch.rand(n, device=device)
            if "x" in edge:
                x = torch.full((n,), coord, device=device, requires_grad=True)
                y = other
            else:
                y = torch.full((n,), coord, device=device, requires_grad=True)
                x = other

            phi = self._predict(x, y, t)
            if "x" in edge:
                dphi_dn = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi),
                                            create_graph=True)[0]
            else:
                dphi_dn = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi),
                                            create_graph=True)[0]
            loss = loss + (dphi_dn**2).mean()
        return loss / 4

    # ── train ────────────────────────────────
    def train(self, epochs: int = 5000, w_pde=1.0, w_ic=10.0, w_bc=1.0,
            print_every=500):
        print("\n" + "="*55)
        print("  Training Allen-Cahn PINN  (ε = {:.3f})".format(self.eps))
        print("="*55)
        t0 = time.time()
        for ep in range(1, epochs + 1):
            self.optimizer.zero_grad()
            l_pde = self._loss_pde()
            l_ic  = self._loss_ic()
            l_bc  = self._loss_bc()
            loss  = w_pde * l_pde + w_ic * l_ic + w_bc * l_bc
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.history["total"].append(loss.item())
            self.history["pde"].append(l_pde.item())
            self.history["ic"].append(l_ic.item())
            self.history["bc"].append(l_bc.item())

            if ep % print_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Ep {ep:5d} | Total {loss.item():.4e} | "
                    f"PDE {l_pde.item():.4e} | IC {l_ic.item():.4e} | "
                    f"BC {l_bc.item():.4e} | lr {lr:.2e}")
        print(f"  Done in {time.time()-t0:.1f}s")

    # ── predict on grid ───────────────────────
    @torch.no_grad()
    def predict_grid(self, t_val: float, nx: int = 80):
        xs = torch.linspace(0, 1, nx, device=device)
        ys = torch.linspace(0, 1, nx, device=device)
        xg, yg = torch.meshgrid(xs, ys, indexing="ij")
        tg = torch.full_like(xg, t_val)
        phi = self._predict(xg.reshape(-1), yg.reshape(-1), tg.reshape(-1))
        return phi.reshape(nx, nx).cpu().numpy()


# ─────────────────────────────────────────────
# 3. CAHN-HILLIARD PINN
# ─────────────────────────────────────────────
class CahnHilliardPINN:
    """
    PDE:  ∂φ/∂t = M Δμ
        μ     = −ε²Δφ + (φ³ − φ)
    IC:   φ(x,y,0) = φ₀(x,y)
    BC:   Neumann (zero-flux) on φ and μ
    """

    def __init__(self, eps: float = 0.05, M: float = 1.0, T: float = 0.5):
        self.eps = eps
        self.M   = M
        self.T   = T
        # Two-output network: [φ, μ]
        self.model = PhaseFieldNet(layers=[3, 128, 128, 128, 128, 2]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9995)
        self.history = {"total": [], "pde1": [], "pde2": [], "ic": [], "bc": []}

    @staticmethod
    def _initial_condition(x, y):
        """Spinodal decomposition-like IC: small random perturbation around 0."""
        torch.manual_seed(7)
        noise = 0.05 * (2 * torch.rand_like(x) - 1)
        return noise

    def _predict(self, x, y, t):
        inp = torch.stack([x, y, t], dim=-1)
        out = self.model(inp)          # (N, 2)
        phi = out[:, 0]
        mu  = out[:, 1]
        return phi, mu

    @staticmethod
    def _laplacian(u, x, y):
        """Compute Δu = ∂²u/∂x² + ∂²u/∂y² with autograd."""
        u_x  = torch.autograd.grad(u,  x, grad_outputs=torch.ones_like(u),
                                    create_graph=True)[0]
        u_y  = torch.autograd.grad(u,  y, grad_outputs=torch.ones_like(u),
                                    create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                    create_graph=True)[0]
        return u_xx + u_yy

    def _pde_residuals(self, x, y, t):
        x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
        phi, mu = self._predict(x, y, t)

        # Eq1: ∂φ/∂t = M Δμ
        phi_t  = torch.autograd.grad(phi, t, grad_outputs=torch.ones_like(phi),
                                    create_graph=True)[0]
        lap_mu = self._laplacian(mu, x, y)
        r1 = phi_t - self.M * lap_mu

        # Eq2: μ = −ε²Δφ + (φ³ − φ)
        lap_phi = self._laplacian(phi, x, y)
        r2 = mu - (-self.eps**2 * lap_phi + phi**3 - phi)

        return r1, r2

    # ── loss terms ───────────────────────────
    def _loss_pde(self, n=2000):
        x = torch.rand(n, device=device)
        y = torch.rand(n, device=device)
        t = torch.rand(n, device=device) * self.T
        r1, r2 = self._pde_residuals(x, y, t)
        return (r1**2).mean(), (r2**2).mean()

    def _loss_ic(self, n=1000):
        x = torch.rand(n, device=device)
        y = torch.rand(n, device=device)
        t = torch.zeros(n, device=device)
        phi_pred, _ = self._predict(x, y, t)
        phi_true = self._initial_condition(x, y)
        return ((phi_pred - phi_true)**2).mean()

    def _loss_bc(self, n=400):
        """Neumann BC: ∂φ/∂n = 0 and ∂μ/∂n = 0 on all walls."""
        loss = torch.tensor(0.0, device=device)
        t = torch.rand(n, device=device) * self.T

        for coord, is_x in [(0.0, True), (1.0, True), (0.0, False), (1.0, False)]:
            other = torch.rand(n, device=device)
            if is_x:
                x = torch.full((n,), coord, device=device, requires_grad=True)
                y = other
            else:
                y = torch.full((n,), coord, device=device, requires_grad=True)
                x = other

            phi, mu = self._predict(x, y, t)
            ref = x if is_x else y

            dphi = torch.autograd.grad(phi, ref, grad_outputs=torch.ones_like(phi),
                                        create_graph=True)[0]
            dmu  = torch.autograd.grad(mu,  ref, grad_outputs=torch.ones_like(mu),
                                        create_graph=True)[0]
            loss = loss + (dphi**2).mean() + (dmu**2).mean()
        return loss / 4

    # ── train ────────────────────────────────
    def train(self, epochs=5000, w_pde=1.0, w_mu=5.0, w_ic=10.0, w_bc=1.0,
            print_every=500):
        print("\n" + "="*55)
        print(f"  Training Cahn-Hilliard PINN  (ε={self.eps}, M={self.M})")
        print("="*55)
        t0 = time.time()
        for ep in range(1, epochs + 1):
            self.optimizer.zero_grad()
            l1, l2 = self._loss_pde()
            l_ic   = self._loss_ic()
            l_bc   = self._loss_bc()
            loss   = w_pde * l1 + w_mu * l2 + w_ic * l_ic + w_bc * l_bc
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.history["total"].append(loss.item())
            self.history["pde1"].append(l1.item())
            self.history["pde2"].append(l2.item())
            self.history["ic"].append(l_ic.item())
            self.history["bc"].append(l_bc.item())

            if ep % print_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Ep {ep:5d} | Total {loss.item():.4e} | "
                    f"φ_t {l1.item():.4e} | μ {l2.item():.4e} | "
                    f"IC {l_ic.item():.4e} | lr {lr:.2e}")
        print(f"  Done in {time.time()-t0:.1f}s")

    @torch.no_grad()
    def predict_grid(self, t_val: float, nx: int = 80):
        xs = torch.linspace(0, 1, nx, device=device)
        ys = torch.linspace(0, 1, nx, device=device)
        xg, yg = torch.meshgrid(xs, ys, indexing="ij")
        tg = torch.full_like(xg, t_val)
        phi, mu = self._predict(xg.reshape(-1), yg.reshape(-1), tg.reshape(-1))
        return (phi.reshape(nx, nx).cpu().numpy(),
                mu.reshape(nx, nx).cpu().numpy())


# ─────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────
def plot_results(ac: AllenCahnPINN, ch: CahnHilliardPINN, save_path: str):
    T = ac.T
    t_snapshots = [0.0, T * 0.25, T * 0.5, T]
    nx = 80

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d0d0d")
    cmap_ac = "RdBu_r"
    cmap_ch = "PiYG"
    cmap_mu = "plasma"

    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.45, wspace=0.35,
                        left=0.06, right=0.97, top=0.92, bottom=0.06)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.7,
                "2D Phase-Field PINNs  ·  Allen-Cahn & Cahn-Hilliard",
                ha="center", va="center", fontsize=17, color="white",
                fontfamily="monospace", fontweight="bold")
    ax_title.text(0.5, 0.15,
                f"ε = {ac.eps}   |   domain [0,1]²   |   T = {T}   "
                f"|   Neumann BC   |   PyTorch PINN",
                ha="center", va="center", fontsize=10, color="#aaaaaa",
                fontfamily="monospace")

    kw_ac = dict(cmap=cmap_ac, vmin=-1.05, vmax=1.05, origin="lower",
                extent=[0, 1, 0, 1])
    kw_ch = dict(cmap=cmap_ch, vmin=-0.15, vmax=0.15, origin="lower",
                extent=[0, 1, 0, 1])

    for col, t_val in enumerate(t_snapshots):
        # Allen-Cahn
        phi_ac = ac.predict_grid(t_val, nx)
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(phi_ac.T, **kw_ac)
        ax.set_title(f"t = {t_val:.3f}", color="#cccccc", fontsize=9,
                    fontfamily="monospace")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        if col == 0:
            ax.set_ylabel("Allen-Cahn  φ", color="#ff9966", fontsize=9,
                        fontfamily="monospace")
        if col == 3:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[-1, 0, 1]).ax.yaxis.set_tick_params(color="white",
                        labelcolor="white")

        # Cahn-Hilliard φ
        phi_ch, mu_ch = ch.predict_grid(t_val, nx)
        ax2 = fig.add_subplot(gs[2, col])
        im2 = ax2.imshow(phi_ch.T, **kw_ch)
        ax2.set_xticks([]); ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")
        if col == 0:
            ax2.set_ylabel("Cahn-Hilliard  φ", color="#66ccff", fontsize=9,
                            fontfamily="monospace")
        if col == 3:
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
                color="white", labelcolor="white")

        # Cahn-Hilliard μ
        ax3 = fig.add_subplot(gs[3, col])
        kw_mu = dict(cmap=cmap_mu, origin="lower", extent=[0, 1, 0, 1])
        im3 = ax3.imshow(mu_ch.T, **kw_mu)
        ax3.set_xticks([]); ax3.set_yticks([])
        for spine in ax3.spines.values():
            spine.set_edgecolor("#444")
        if col == 0:
            ax3.set_ylabel("Cahn-Hilliard  μ", color="#cc99ff", fontsize=9,
                            fontfamily="monospace")
        if col == 3:
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
                color="white", labelcolor="white")

    # Loss curves
    ax_l1 = fig.add_subplot(gs[1, 4:])
    ax_l2 = fig.add_subplot(gs[2, 4:])
    ax_l3 = fig.add_subplot(gs[3, 4:])

    def _plot_loss(ax, history, keys, colors, title):
        for k, c in zip(keys, colors):
            ax.semilogy(history[k], color=c, lw=1.2, label=k)
        ax.set_facecolor("#1a1a1a")
        ax.set_title(title, color="#cccccc", fontsize=9, fontfamily="monospace")
        ax.tick_params(colors="#888")
        ax.spines[:].set_color("#444")
        ax.legend(fontsize=7, labelcolor="white", facecolor="#111")
        ax.set_xlabel("epoch", color="#888", fontsize=8)

    _plot_loss(ax_l1, ac.history,
               ["total", "pde", "ic", "bc"],
            ["#ff9966", "#ff4444", "#44ff88", "#4488ff"],
            "Allen-Cahn losses")
    _plot_loss(ax_l2, ch.history,
               ["total", "pde1", "pde2", "ic"],
            ["#66ccff", "#ff4444", "#ffaa44", "#44ff88"],
            "Cahn-Hilliard losses")

    # Free energy over time (Allen-Cahn bulk energy estimate)
    t_vals = np.linspace(0, T, 20)
    energies_ac, mass_ch = [], []
    for tv in t_vals:
        p = ac.predict_grid(tv, 40)
        h = 1.0 / 39
        energies_ac.append(h**2 * np.sum((p**2 - 1)**2 / 4))

        pc, _ = ch.predict_grid(tv, 40)
        mass_ch.append(pc.mean())

    ax_l3.set_facecolor("#1a1a1a")
    ax_l3_r = ax_l3.twinx()
    ax_l3.plot(t_vals, energies_ac, color="#ff9966", lw=1.5, label="AC bulk energy")
    ax_l3_r.plot(t_vals, mass_ch, color="#66ccff", lw=1.5, linestyle="--",
                label="CH mass (conserved)")
    ax_l3.set_title("Physics diagnostics", color="#cccccc", fontsize=9,
                    fontfamily="monospace")
    ax_l3.tick_params(colors="#888"); ax_l3_r.tick_params(colors="#888")
    ax_l3.spines[:].set_color("#444"); ax_l3_r.spines[:].set_color("#444")
    ax_l3.set_xlabel("t", color="#888", fontsize=8)
    ax_l3.legend(fontsize=7, labelcolor="white", facecolor="#111", loc="upper left")
    ax_l3_r.legend(fontsize=7, labelcolor="white", facecolor="#111", loc="upper right")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Figure saved → {save_path}")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    EPOCHS = 3000   # increase to 8000+ for publication quality
    T      = 0.3
    EPS    = 0.05

    ac = AllenCahnPINN(eps=EPS, T=T)
    ac.train(epochs=EPOCHS, print_every=500)

    ch = CahnHilliardPINN(eps=EPS, M=1.0, T=T)
    ch.train(epochs=EPOCHS, print_every=500)

    out = "phase_field_pinns.png"
    # Save the trained models first so you don't lose them
    torch.save(ac.model.state_dict(), "ac_model.pth")
    torch.save(ch.model.state_dict(), "ch_model.pth")

    out = "phase_field_pinns.png"   # ← fixed path
    plot_results(ac, ch, out)
    print("\nAll done.")
