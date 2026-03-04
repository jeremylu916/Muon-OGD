import torch
from torch.optim import Optimizer


@torch.no_grad()
def _polar_newton_schulz(X: torch.Tensor, iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    dev = X.device
    orig_dtype = X.dtype
    A = X.detach().to(torch.float32).to(dev)
    m, n = A.shape
    frob = torch.linalg.norm(A, ord="fro")
    if frob < eps:
        return torch.zeros_like(X)
    A = A / (frob + eps)

    if m >= n:
        Y = A
        I = torch.eye(n, device=dev, dtype=torch.float32)
        Z = I.clone()
        for _ in range(iters):
            ZY = Z @ Y.transpose(0, 1) @ Y
            T = 0.5 * (3.0 * I - ZY)
            Y = Y @ T
            Z = T @ Z
        P = Y
    else:
        At = A.transpose(0, 1)
        Y = At
        I = torch.eye(m, device=dev, dtype=torch.float32)
        Z = I.clone()
        for _ in range(iters):
            ZY = Z @ Y.transpose(0, 1) @ Y
            T = 0.5 * (3.0 * I - ZY)
            Y = Y @ T
            Z = T @ Z
        P = Y.transpose(0, 1)

    return P.to(dev).to(orig_dtype)


def _matrix_sign_via_svd(X: torch.Tensor) -> torch.Tensor:
    dev = X.device
    orig_dtype = X.dtype
    Xf = X.detach().to(torch.float32).to(dev)
    U, _, Vh = torch.linalg.svd(Xf, full_matrices=False)
    return (U @ Vh).to(dev).to(orig_dtype)


def _msgn(X: torch.Tensor, method: str = "ns", ns_iters: int = 6) -> torch.Tensor:
    if method == "svd":
        return _matrix_sign_via_svd(X)
    return _polar_newton_schulz(X, iters=ns_iters)


def _rank1_inner_products(Cs, S_mat_fp32: torch.Tensor) -> torch.Tensor:
    if not Cs:
        return torch.zeros(0, device=S_mat_fp32.device, dtype=torch.float32)
    vals = []
    for c in Cs:
        vals.append(c.sigma * (c.u @ (S_mat_fp32 @ c.v)))
    return torch.stack(vals, dim=0)


def _add_rank1_shift_(H_fp32: torch.Tensor, Cs, lam_fp32: torch.Tensor) -> torch.Tensor:
    for i, c in enumerate(Cs):
        alpha = lam_fp32[i] * c.sigma
        H_fp32.add_(alpha * (c.u.unsqueeze(1) @ c.v.unsqueeze(0)))
    return H_fp32


class MuonOGDOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-4,
        momentum=0.0,
        weight_decay=0.0,
        muon_T=1,
        muon_eta_dual=1e-4,
        msign_method="ns",
        ns_iters=6,
        dynamic_scale=False,
        warm_start=True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            muon_T=muon_T,
            muon_eta_dual=muon_eta_dual,
            msign_method=msign_method,
            ns_iters=ns_iters,
            dynamic_scale=dynamic_scale,
            warm_start=warm_start,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            muon_T = group["muon_T"]
            muon_eta_dual = group["muon_eta_dual"]
            msign_method = group["msign_method"]
            ns_iters = group["ns_iters"]
            dynamic_scale = bool(group["dynamic_scale"])
            warm_start = bool(group["warm_start"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise RuntimeError("MuonOGDOptimizer only supports 2D parameters.")

                grad = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad, dtype=torch.float32)
                if "lam" not in state:
                    state["lam"] = None

                buf = state["momentum_buffer"]
                grad_fp32 = grad.detach().to(torch.float32)
                if momentum > 0.0:
                    buf.mul_(momentum).add_(grad_fp32, alpha=(1.0 - momentum))
                else:
                    buf.copy_(grad_fp32)

                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                Cs = state.get("Cs", [])
                k = len(Cs)

                if k > 0 and (state["lam"] is None or state["lam"].numel() != k):
                    state["lam"] = torch.zeros(k, dtype=torch.float32, device=p.device)

                if k == 0:
                    S_final = _msgn(buf, method=msign_method, ns_iters=ns_iters).to(torch.float32)
                else:
                    if warm_start:
                        lam = state["lam"]
                    else:
                        lam = torch.zeros(k, dtype=torch.float32, device=p.device)

                    for _ in range(muon_T):
                        H = buf.clone()
                        _add_rank1_shift_(H, Cs, lam)
                        S = _msgn(H, method=msign_method, ns_iters=ns_iters).to(torch.float32)
                        inner = _rank1_inner_products(Cs, S)
                        lam.sub_(muon_eta_dual * inner)

                    H_final = buf.clone()
                    _add_rank1_shift_(H_final, Cs, lam)
                    S_final = _msgn(H_final, method=msign_method, ns_iters=ns_iters).to(torch.float32)

                    if warm_start:
                        state["lam"] = lam

                if dynamic_scale:
                    scale = torch.clamp(torch.linalg.norm(p.detach().to(torch.float32)), min=1e-3)
                    update = S_final.to(p.dtype) * (lr * scale.to(p.dtype))
                else:
                    update = S_final.to(p.dtype) * lr
                p.sub_(update)

        return loss
