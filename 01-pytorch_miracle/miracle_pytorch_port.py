# MIRACLE model (TensorFlow v1) → PyTorch port
#
# Notes
# - Faithful port of architecture and losses from the original TF code.
# - Keeps per-target input layer (W_h0_i) and output head (out_i) with shared hidden (w_h1).
# - Replicates indicator masking, group-lasso penalty, truncated power-series DAG penalty, and
#   the supervised + BCE objective on observed entries / indicators.
# - Preserves the imputation loop with a sliding window average of recent predictions.
#
# API differences
# - Uses a standard PyTorch nn.Module with a `.fit(...)` method that mirrors the TF class.
# - No sessions/placeholder; pass tensors/ndarrays directly.
# - Checkpointing is optional and not enabled by default (add torch.save if needed).
#
# Dependencies: torch>=2.0, numpy, pandas, scikit-learn (for MSE if you want parity)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiracleConfig:
    lr: float = 0.001
    batch_size: int = 32
    num_inputs: int = 1  # will be overwritten after indicators are appended
    num_outputs: int = 1
    n_hidden: int = 32
    reg_lambda: float = 1.0
    reg_beta: float = 1.0
    reg_m: float = 1.0
    DAG_only: bool = False
    window: int = 10
    max_steps: int = 400
    random_seed: int = 0
    device: str = "cpu"  # or "cuda"
    verbose: bool = False


class MiracleTorch(nn.Module):
    def __init__(self, config: MiracleConfig, n_indicators: int, missing_list: List[int]):
        super().__init__()
        torch.manual_seed(config.random_seed)
        self.cfg = config
        self.n_indicators = int(n_indicators)
        self.missing_list = list(missing_list)
        
        if self.cfg.verbose:
            print(f"[MIRACLE] Initializing model with {n_indicators} indicators for missing columns: {missing_list}")
            print(f"[MIRACLE] Device: {config.device}, Hidden units: {config.n_hidden}")
            print(f"[MIRACLE] Regularization - lambda: {config.reg_lambda}, beta: {config.reg_beta}, m: {config.reg_m}")

        # Effective input count: original features + indicator columns
        self.num_inputs = config.num_inputs + self.n_indicators
        self.num_outputs = config.num_outputs
        self.n_hidden = config.n_hidden

        # Per-target input layer and output head
        self.W_h0: nn.ParameterDict = nn.ParameterDict()
        self.b_h0: nn.ParameterDict = nn.ParameterDict()
        self.W_out: nn.ParameterDict = nn.ParameterDict()
        self.b_out: nn.ParameterDict = nn.ParameterDict()

        # Xavier/Glorot initialization for better stability
        for i in range(self.num_inputs):
            w_h0 = torch.randn(self.num_inputs, self.n_hidden)
            nn.init.xavier_uniform_(w_h0)
            self.W_h0[f"w_h0_{i}"] = nn.Parameter(w_h0)
            self.b_h0[f"b_h0_{i}"] = nn.Parameter(torch.zeros(self.n_hidden))
            
            w_out = torch.randn(self.n_hidden, self.num_outputs)
            nn.init.xavier_uniform_(w_out)
            self.W_out[f"out_{i}"] = nn.Parameter(w_out)
            self.b_out[f"out_{i}"] = nn.Parameter(torch.zeros(self.num_outputs))

        # Shared hidden layer
        w_h1 = torch.randn(self.n_hidden, self.n_hidden)
        nn.init.xavier_uniform_(w_h1)
        self.W_h1 = nn.Parameter(w_h1)
        self.b_h1 = nn.Parameter(torch.zeros(self.n_hidden))

        # Precompute the indicator mask and per-target exclusion masks as buffers (no grad)
        # Masks zero specific columns in W_h0 so a target i cannot see its own column (and indicator columns
        # are mutually excluded as in the TF code).
        with torch.no_grad():
            # Build indicator exclusion mask
            if self.n_indicators > 0:
                # Start by excluding the first indicator anchor column: index = num_inputs - n_indicators
                indicator_mask = F.one_hot(
                    torch.full((self.n_hidden,), self.num_inputs - self.n_indicators, dtype=torch.long),
                    num_classes=self.num_inputs,
                ).float().t()
                indicator_mask = 1.0 - indicator_mask  # on_value=0, off=1
                # Also exclude the rest indicator columns
                for i in range(self.num_inputs - self.n_indicators + 1, self.num_inputs):
                    m = F.one_hot(torch.full((self.n_hidden,), i, dtype=torch.long), self.num_inputs).float().t()
                    indicator_mask = indicator_mask * (1.0 - m)
            else:
                # No indicators, allow all connections
                indicator_mask = torch.ones(self.num_inputs, self.n_hidden)

            # Per-target exclusion masks
            masks: Dict[str, torch.Tensor] = {}
            for i in range(self.num_inputs):
                m = F.one_hot(torch.full((self.n_hidden,), i, dtype=torch.long), self.num_inputs).float().t()
                masks[str(i)] = (1.0 - m) * indicator_mask  # (num_inputs, n_hidden)

        # Register as buffers for easy device transfer and to ensure they’re saved with the model
        self.register_buffer("indicator_mask", indicator_mask)
        for k, v in masks.items():
            self.register_buffer(f"mask_{k}", v)

        self.activation = F.elu
        self.to(self.cfg.device)

    # ---- Core forward primitives -------------------------------------------------
    def _masked_linear0(self, X: torch.Tensor, i: int) -> torch.Tensor:
        # Apply per-target mask to the first-layer weights
        W = getattr(self, f"mask_{i}") * self.W_h0[f"w_h0_{i}"]
        return X @ W + self.b_h0[f"b_h0_{i}"]

    def _forward_per_target(self, X: torch.Tensor, i: int) -> torch.Tensor:
        h0 = self.activation(self._masked_linear0(X, i))
        h1 = self.activation(h0 @ self.W_h1 + self.b_h1)
        out = h1 @ self.W_out[f"out_{i}"] + self.b_out[f"out_{i}"]
        return out  # (N, num_outputs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for i in range(self.num_inputs):
            outs.append(self._forward_per_target(X, i))
        return torch.cat(outs, dim=1)  # (N, num_inputs) since num_outputs==1

    # ---- Regularizers & penalties ------------------------------------------------
    def group_lasso(self) -> torch.Tensor:
        """Replicates TF group-lasso over rows of masked W_h0_i.
        For non-indicator targets i < num_inputs - n_indicators:
          sum(||rows before i||_2) + sum(||rows after i up to non-indicator count||_2)
        For indicator targets: use rows over the non-indicator block only.
        """
        total = torch.tensor(0.0, device=self.cfg.device)
        non_ind_cnt = self.num_inputs - self.n_indicators
        
        if non_ind_cnt <= 0:  # Edge case: all columns are indicators
            return total
            
        for i in range(self.num_inputs):
            W = getattr(self, f"mask_{i}") * self.W_h0[f"w_h0_{i}"]  # (num_inputs, n_hidden)
            if i < non_ind_cnt:
                if i > 0:
                    w1 = W[:i, :]  # rows before i
                    total = total + w1.norm(dim=1).sum()
                if i + 1 < non_ind_cnt:
                    w2 = W[i + 1 : non_ind_cnt, :]  # rows after i within non-indicator block
                    total = total + w2.norm(dim=1).sum()
            else:
                if non_ind_cnt > 0:
                    w1 = W[:non_ind_cnt, :]
                    total = total + w1.norm(dim=1).sum()
        return total

    def dag_penalty(self) -> torch.Tensor:
        """Truncated power-series approximation from the TF code.
        Builds W (non-negative) as sqrt(sum of squares across hidden units) for every (source->target),
        then computes h = sum_{k>=1} trace(Z^k)/k! with Z=W∘W (elementwise square), up to k=24, minus d.
        """
        # Build W: concat over targets of sqrt(sum_j W_ij^2)
        rows: List[torch.Tensor] = []
        for i in range(self.num_inputs):
            W = getattr(self, f"mask_{i}") * self.W_h0[f"w_h0_{i}"]  # (num_inputs, n_hidden)
            row = (W.pow(2).sum(dim=1, keepdim=True) + 1e-8).sqrt()  # Add small epsilon for stability
            rows.append(row)
        Wmat = torch.cat(rows, dim=1)  # (num_inputs, num_inputs)
        Z = Wmat.pow(2)  # elementwise square
        d = float(self.num_inputs)

        I = torch.eye(self.num_inputs, device=self.cfg.device)
        Z_in = I.clone()
        dag_l = torch.tensor(d, device=self.cfg.device)
        coff = 1.0
        
        # Limit to fewer iterations to prevent overflow
        max_k = min(15, self.num_inputs * 2)  # Reduce from 24 to prevent overflow
        for k in range(1, max_k + 1):
            Z_in = Z_in @ Z
            trace_val = torch.trace(Z_in)
            # Add numerical stability check
            if torch.isnan(trace_val) or torch.isinf(trace_val):
                break
            dag_l = dag_l + trace_val / coff
            coff *= (k + 1)
        h = dag_l - d
        return torch.clamp(h, max=1e6)  # Prevent extremely large values

    # ---- Training / Imputation loop ---------------------------------------------
    @torch.no_grad()
    def _transform(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.cfg.device)
        Out = self.forward(X_t)
        return Out.cpu().numpy()

    def _build_subset_mask(self, num_cols: int) -> np.ndarray:
        # Mimic TF: choose a random subset of columns of size = num_cols (here effectively ~ all)
        one_hot = [0] * self.num_inputs
        subset = random.sample(range(self.num_inputs), num_cols)
        for j in subset:
            one_hot[j] = 1
        return np.array(one_hot, dtype=np.int64)

    def _fit(
        self,
        X_missing_c: np.ndarray,
        X_mask_c: np.ndarray,
        X_seed_c: Optional[np.ndarray] = None,
        early_stopping: bool = False,
        reg_lambda_val: float = 1.0,
        rho_val: float = 1.0,
        alpha_val: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        device = self.cfg.device
        self.train()

        X = X_missing_c.copy()
        N, D = X.shape
        assert D == self.num_inputs, "Shape mismatch after concatenating indicators"

        # Init X if no seed provided
        if X_seed_c is None:
            X_seed_c = X.copy()
            # Handle mean computation more carefully
            col_mean = np.nanmean(X_seed_c, axis=0)
            # Replace NaNs in col_mean with zeros if entire column is missing
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            inds = np.where(np.isnan(X_seed_c))
            X_seed_c[inds] = np.take(col_mean, inds[1])
        X = X_seed_c.copy()
        
        # Ensure no NaN values remain
        if np.any(np.isnan(X)):
            if self.cfg.verbose:
                print("[MIRACLE] Warning: NaN values found after initialization, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0)

        # Optimizer
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

        # Subset training columns mask
        subset_one_hot = self._build_subset_mask(num_cols=D)
        subset_idx = np.where(subset_one_hot == 1)[0]

        # Sliding window of predictions
        avg_seed: List[np.ndarray] = []
        best_loss = math.inf
        best_state = None
        
        if self.cfg.verbose:
            print(f"[MIRACLE] Starting training for {self.cfg.max_steps} steps")
            print(f"[MIRACLE] Data shape: {X.shape}, Missing columns: {self.missing_list}")
            print(f"[MIRACLE] Batch size: {self.cfg.batch_size}, Window size: {self.cfg.window}")

        for step in range(1, self.cfg.max_steps):
            # After warm-up/window, refresh missing entries with avg of recent predictions
            if step > self.cfg.window and len(avg_seed) > 0:
                X_pred = np.mean(np.stack(avg_seed, axis=0), axis=0)
                X = X * X_mask_c + X_pred * (1 - X_mask_c)

            # One full pass in mini-batches
            idxs = np.arange(N)
            np.random.shuffle(idxs)
            for start in range(0, N, self.cfg.batch_size):
                bs = slice(start, min(start + self.cfg.batch_size, N))
                xb = torch.as_tensor(X[idxs[bs]], dtype=torch.float32, device=device)
                mb = torch.as_tensor(X_mask_c[idxs[bs]], dtype=torch.float32, device=device)

                Out = self.forward(xb)  # (B, D)
                R = xb - Out

                # Supervised squared error on non-indicator block with mask
                non_ind = D - self.n_indicators
                R_proxy = ((xb[:, :non_ind] * mb[:, :non_ind]) - (Out[:, :non_ind] * mb[:, :non_ind])) ** 2
                supervised_loss = R_proxy.sum()

                # BCE on indicator logits vs observed indicator labels (present=0/1 constructed below)
                if self.n_indicators > 0:
                    bce = F.binary_cross_entropy_with_logits(
                        input=Out[:, non_ind :],
                        target=xb[:, non_ind :],
                        reduction="none",
                    )
                    R_proxy = torch.cat([R_proxy, bce], dim=1)
                else:
                    bce = torch.tensor(0.0, device=device)

                # Group lasso
                L1 = self.group_lasso()

                # DAG penalty
                h = self.dag_penalty()

                # Moment regularizer over indicators
                reg_moment = torch.tensor(0.0, device=device)
                for dim in range(self.n_indicators):
                    index = self.missing_list[dim]
                    probs = torch.sigmoid(Out[:, non_ind + dim])
                    probs = torch.clamp((probs + 1.01) / 1.02, min=1e-6, max=1-1e-6)  # Prevent division by zero
                    weight = 1.0 / probs
                    a = Out[:, index].mean()
                    
                    # Add numerical stability to the weighted sum
                    weighted_sum = (xb[:, index] * mb[:, index]) * weight
                    weight_sum = weight.sum()
                    if weight_sum > 1e-8:  # Only compute if weights are meaningful
                        b = weighted_sum.sum() / weight_sum
                    else:
                        b = a  # Fallback to avoid NaN
                    reg_moment = reg_moment + (a - b).pow(2)

                # Select only columns in subset for MSE aggregation (like TF dynamic_partition)
                subset_R = R_proxy[:, subset_idx]
                mse_loss_subset = subset_R.sum(dim=1).mean()

                reg_loss_subset = (
                    mse_loss_subset
                    + self.cfg.reg_beta * L1
                    + 0.5 * rho_val * (h * h)
                    + alpha_val * h
                    + self.cfg.reg_m * reg_moment
                )

                if not self.cfg.DAG_only:
                    reg_loss_subset = reg_loss_subset + reg_lambda_val * rho_val * supervised_loss

                # Check for NaN loss and skip if found
                if torch.isnan(reg_loss_subset) or torch.isinf(reg_loss_subset):
                    if self.cfg.verbose:
                        print(f"[MIRACLE] Warning: NaN/Inf loss detected at step {step}, skipping update")
                    continue

                opt.zero_grad()
                reg_loss_subset.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                opt.step()

            # Track average predictions
            with torch.no_grad():
                X_pred = self.forward(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
            if len(avg_seed) >= self.cfg.window:
                avg_seed.pop(0)
            avg_seed.append(X_pred)

            # Simple early-stopping by supervised loss on full data (optional)
            with torch.no_grad():
                xb = torch.as_tensor(X, dtype=torch.float32, device=device)
                mb = torch.as_tensor(X_mask_c, dtype=torch.float32, device=device)
                Out = self.forward(xb)
                non_ind = D - self.n_indicators
                R_proxy_full = ((xb[:, :non_ind] * mb[:, :non_ind]) - (Out[:, :non_ind] * mb[:, :non_ind])) ** 2
                loss_val = R_proxy_full.sum().item()
                
            if self.cfg.verbose and (step % 25 == 0 or step == 1):
                print(f"[MIRACLE] Step {step}/{self.cfg.max_steps}, Loss: {loss_val:.6f}")
                if step == 1:
                    print(f"  - Supervised loss: {supervised_loss.item():.6f}")
                    print(f"  - Group lasso: {L1.item():.6f}")
                    print(f"  - DAG penalty: {h.item():.6f}")
                    print(f"  - Moment reg: {reg_moment.item():.6f}")
                
            if early_stopping and step > 10 and loss_val < best_loss:
                best_loss = loss_val
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                if self.cfg.verbose:
                    print(f"[MIRACLE] New best loss: {best_loss:.6f} at step {step}")

        if early_stopping and best_state is not None:
            self.load_state_dict(best_state)

        # Final blended output
        X_pred = np.mean(np.stack(avg_seed, axis=0), axis=0) if len(avg_seed) > 0 else X
        X_final = X * X_mask_c + X_pred * (1 - X_mask_c)
        return X_final, X_pred, avg_seed

    # ---- Public fit API (mirrors TF wrapper) ------------------------------------
    def fit(
        self,
        X_missing: np.ndarray,
        missing_list: List[int],
        X_seed: Optional[np.ndarray] = None,
        early_stopping: bool = False,
    ) -> np.ndarray:
        """
        X_missing: (N, P) with NaNs marking missing entries in the original features.
        missing_list: indices (in original P space) of columns with any missingness; used for reg_moment.
        Returns imputed array with shape (N, P).
        """
        N, P = X_missing.shape
        assert P + self.n_indicators == self.num_inputs, "Model must be constructed with matching indicator count"
        
        if self.cfg.verbose:
            missing_count = np.isnan(X_missing).sum()
            missing_rate = missing_count / (N * P)
            print(f"[MIRACLE] Input data: {N} samples x {P} features")
            print(f"[MIRACLE] Missing values: {missing_count} ({missing_rate:.2%})")

        # Build indicator matrix from NaN pattern (1 if observed, 0 if missing) then invert (TF used 1 for missing in df_mask, then 1 - indicators)
        df_mask = np.isnan(X_missing).astype(float)
        ind_all = 1.0 - df_mask  # 1 if observed, 0 if missing
        # Use indicators ONLY for columns that actually have missingness
        if len(missing_list) > 0:
            indicators = ind_all[:, missing_list]
        else:
            indicators = np.zeros((N, 0), dtype=float)
            
        if self.cfg.verbose:
            print(f"[MIRACLE] Created {indicators.shape[1]} indicator columns for missing features")

        X_MASK = np.ones_like(X_missing, dtype=float)
        X_MASK[np.isnan(X_missing)] = 0.0

        X_MISSING_c = np.concatenate([X_missing, indicators], axis=1)
        X_seed_c = None
        if X_seed is not None:
            X_seed_c = np.concatenate([X_seed, indicators], axis=1)
        X_MASK_c = np.concatenate([X_MASK, np.ones((N, indicators.shape[1]))], axis=1)

        X_filled, _, _ = self._fit(
            X_MISSING_c, X_MASK_c, X_seed_c, early_stopping=early_stopping,
            reg_lambda_val=self.cfg.reg_lambda, rho_val=1.0, alpha_val=1.0,
        )
        
        if self.cfg.verbose:
            print(f"[MIRACLE] Imputation completed. Output shape: {X_filled[:, :P].shape}")
            
        return X_filled[:, :P]


# ---------------------------- Convenience wrapper --------------------------------

def miracle_impute(
    X_missing: np.ndarray,
    missing_list: List[int],
    lr: float = 0.001,
    batch_size: int = 32,
    n_hidden: int = 32,
    reg_lambda: float = 1.0,
    reg_beta: float = 1.0,
    reg_m: float = 1.0,
    DAG_only: bool = False,
    window: int = 10,
    max_steps: int = 400,
    random_seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> np.ndarray:
    """One-shot helper mirroring the TF class behavior."""
    N, P = X_missing.shape
    cfg = MiracleConfig(
        lr=lr,
        batch_size=batch_size,
        num_inputs=P,  # original P; model adds indicators internally via constructor
        num_outputs=1,
        n_hidden=n_hidden,
        reg_lambda=reg_lambda,
        reg_beta=reg_beta,
        reg_m=reg_m,
        DAG_only=DAG_only,
        window=window,
        max_steps=max_steps,
        random_seed=random_seed,
        device=device,
        verbose=verbose,
    )

    # Number of indicator columns equals count of columns containing any NaNs
    n_indicators = len(missing_list)
    model = MiracleTorch(cfg, n_indicators=n_indicators, missing_list=missing_list)
    return model.fit(X_missing, missing_list=missing_list, X_seed=None, early_stopping=False)


