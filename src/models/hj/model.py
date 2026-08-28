"""Drop-in UNSB subclass that adds the clean HJ-PatchNCE structure projection."""

from __future__ import annotations

import torch

from .core import (
    StructureProjectConfig,
    finite_step_window_active,
    structure_project_nce_step,
)
from .diagnostics import EpochDiagnostics
from models.sb_model import SBModel


class SBModelHJPatchNCE(SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--hj_enable", type=util_str2bool, default=False)
        parser.add_argument("--hj_layers", type=str, default="0")
        parser.add_argument("--hj_direction", type=str, default="joint")
        parser.add_argument("--hj_scales", type=str, default="1,2,4")
        parser.add_argument("--hj_step", type=float, default=0.01)
        parser.add_argument("--hj_quantile", type=float, default=0.75)
        parser.add_argument("--hj_gate_quantile", type=float, default=0.75)
        parser.add_argument("--hj_strength", type=float, default=0.5)
        parser.add_argument("--hj_boundary_scale", type=float, default=0.001)
        parser.add_argument("--hj_min_risk", type=float, default=0.05)
        parser.add_argument("--hj_min_delta", type=float, default=0.0)
        parser.add_argument("--hj_probe_mode", type=str, default="central_consensus")
        parser.add_argument("--hj_control", type=str, default="true")
        parser.add_argument("--hj_amplitude", type=str, default="constant")
        parser.add_argument("--hj_update_mode", type=str, default="remove")
        parser.add_argument("--hj_start_epoch", type=int, default=5)
        parser.add_argument("--hj_direction_alpha", type=float, default=0.0,
                            help="0=pure structure direction, 1=pure fixed random direction")
        parser.add_argument("--hj_random_seed", type=int, default=2026)
        parser.add_argument("--hj_schedule", type=str, default="constant")
        parser.add_argument("--hj_diag_out", type=str, default="")
        parser.add_argument(
            "--hj_search_start_step", type=int, default=-1,
            help="search-runner override for the first active optimizer step",
        )
        parser.add_argument(
            "--hj_search_duration_steps", type=int, default=0,
            help="finite HJ steering duration; <=0 keeps the historical open-ended window",
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.hj_layers = [int(x) for x in str(getattr(opt, "hj_layers", "0")).split(",")]
        self.hj_config = StructureProjectConfig(
            direction=str(getattr(opt, "hj_direction", "joint")),
            scales=str(getattr(opt, "hj_scales", "1,2,4")),
            step=float(getattr(opt, "hj_step", 0.01)),
            quantile=float(getattr(opt, "hj_quantile", 0.75)),
            gate_quantile=float(getattr(opt, "hj_gate_quantile", 0.75)),
            strength=float(getattr(opt, "hj_strength", 0.5)),
            boundary_scale=float(getattr(opt, "hj_boundary_scale", 0.001)),
            min_risk=float(getattr(opt, "hj_min_risk", 0.05)),
            min_delta=float(getattr(opt, "hj_min_delta", 0.0)),
            probe_mode=str(getattr(opt, "hj_probe_mode", "central_consensus")),
            control=str(getattr(opt, "hj_control", "true")),
            amplitude=str(getattr(opt, "hj_amplitude", "constant")),
            update_mode=str(getattr(opt, "hj_update_mode", "remove")),
            start_epoch=int(getattr(opt, "hj_start_epoch", 5)),
            direction_alpha=float(getattr(opt, "hj_direction_alpha", 0.0)),
            random_seed=int(getattr(opt, "hj_random_seed", 2026)),
        )
        self.hj_epoch = 0
        self._hj_step_in_epoch = 0
        self._hj_gate_sum = 0.0
        self._hj_risk_sum = 0.0
        self._hj_probe_sum = 0.0
        self._hj_risk_positive_sum = 0.0
        self._hj_sb_grad_norm = 0.0
        self._hj_conflict_ema = None
        self._hj_conflict_peak = 0.0
        self._hj_adaptive_weight = 1.0
        # Read-only engineering diagnostic: proves that a smoke/production run
        # crossed the configured HJ window. It never participates in a loss.
        self._hj_active_optimizer_steps = 0
        self._hj_diag = (
            EpochDiagnostics(getattr(opt, "hj_diag_out", ""))
            if getattr(opt, "hj_diag_out", "")
            else None
        )

    def _hj_active(self):
        search_start = int(getattr(self.opt, "hj_search_start_step", -1))
        if search_start >= 0:
            time_ready = finite_step_window_active(
                int(getattr(self, "_search_global_step", 0)),
                search_start,
                int(getattr(self.opt, "hj_search_duration_steps", 0)),
            )
        else:
            time_ready = self.hj_epoch >= int(getattr(self.opt, "hj_start_epoch", 5))
        return (
            self.isTrain
            and bool(getattr(self.opt, "hj_enable", False))
            and self.opt.lambda_NCE > 0.0
            and time_ready
        )

    def _hj_probe_fn(self, z, layer, sample_ids):
        def probe(tgt_img):
            # Must use the full nce_layers list: netG's encode_only early-return
            # is keyed on layers[-1], so a single-element list returns the wrong
            # structure (a tuple instead of a feature list).
            fq = self.netG(tgt_img, self.time_idx * 0, z, self.nce_layers, encode_only=True)
            if self.opt.flip_equivariance and self.flipped_for_equivariance:
                fq = [torch.flip(f, [3]) for f in fq]
            fq_pool, _ = self.netF(fq, self.opt.num_patches, sample_ids)
            idx = self.nce_layers.index(int(layer))
            return fq_pool[idx]
        return probe

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        # Preserve the canonical RNG stream exactly while HJ is inactive.
        # ``torch.randn(..., device=cuda)`` is not stream-equivalent to the
        # original CPU draw followed by ``.to(device)``.
        z = torch.randn(size=[self.real_A.size(0), 4 * self.opt.ngf]).to(
            self.real_A.device
        )
        feat_q = self.netG(tgt, self.time_idx * 0, z, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = self.netG(src, self.time_idx * 0, z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total = 0.0
        for f_q, f_k, crit, nce_layer in zip(
            feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers
        ):
            if self._hj_active() and nce_layer in self.hj_layers:
                loss, diag = structure_project_nce_step(
                    feat_q=f_q,
                    feat_k=f_k,
                    criterion=crit,
                    source=src,
                    tgt_nce=tgt,
                    probe_fn=self._hj_probe_fn(z, nce_layer, sample_ids),
                    batch_size=self.real_A.size(0),
                    cfg=self.hj_config,
                    lambda_nce=self.opt.lambda_NCE,
                    schedule_weight=self._hj_schedule_weight(),
                )
                total += loss
                if self._hj_diag is not None:
                    self._hj_gate_sum += float(diag["gate_active"])
                    self._hj_risk_sum += float(diag["risk_mean"])
                    self._hj_probe_sum += float(diag["probe_agreement"])
                    self._hj_risk_positive_sum += float(diag["risk_positive"])
            else:
                total += (crit(f_q, f_k) * self.opt.lambda_NCE).mean()
        return total / n_layers

    def _hj_schedule_weight(self) -> float:
        if str(getattr(self.opt, "hj_schedule", "constant")) != "adaptive":
            return 1.0
        return self._hj_adaptive_weight

    def compute_G_loss(self):
        active_this_step = self._hj_active()
        base_loss = super().compute_G_loss()
        if active_this_step:
            self._hj_active_optimizer_steps += 1
        self._hj_step_in_epoch += 1
        if self._hj_step_in_epoch == 1 and self._hj_diag is not None:
            loss = getattr(self, "loss_SB", None)
            if torch.is_tensor(loss):
                try:
                    grads = torch.autograd.grad(
                        loss,
                        list(self.netG.parameters()),
                        retain_graph=True,
                        allow_unused=True,
                    )
                    total = 0.0
                    for g in grads:
                        if g is not None:
                            total += float((g.detach() ** 2).sum().item())
                    self._hj_sb_grad_norm = total ** 0.5
                except Exception:
                    self._hj_sb_grad_norm = 0.0
        return base_loss

    def update_learning_rate(self):
        super().update_learning_rate()
        self.hj_epoch += 1
        if self._hj_diag is not None:
            n = max(self._hj_step_in_epoch, 1)
            self._hj_diag.log(
                epoch=self.hj_epoch,
                gate_hit_rate=self._hj_gate_sum / n,
                risk_mean=self._hj_risk_sum / n,
                probe_agreement=self._hj_probe_sum / n,
                risk_positive=self._hj_risk_positive_sum / n,
                sb_entropy_grad_norm=self._hj_sb_grad_norm,
            )
        if str(getattr(self.opt, "hj_schedule", "constant")) == "adaptive":
            conflict = self._hj_gate_sum / max(self._hj_step_in_epoch, 1)
            if self._hj_conflict_ema is None:
                self._hj_conflict_ema = conflict
            else:
                self._hj_conflict_ema = 0.9 * self._hj_conflict_ema + 0.1 * conflict
            self._hj_conflict_peak = max(self._hj_conflict_peak, self._hj_conflict_ema)
            self._hj_adaptive_weight = (
                self._hj_conflict_ema / max(self._hj_conflict_peak, 1e-9)
            )
        self._hj_step_in_epoch = 0
        self._hj_gate_sum = 0.0
        self._hj_risk_sum = 0.0
        self._hj_probe_sum = 0.0
        self._hj_risk_positive_sum = 0.0

    def set_train_epoch(self, epoch):
        """Set the physical epoch used by the HJ start gate."""
        self.hj_epoch = int(epoch)

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        names = (
            "hj_epoch", "_hj_step_in_epoch", "_hj_gate_sum",
            "_hj_risk_sum", "_hj_probe_sum", "_hj_risk_positive_sum",
            "_hj_sb_grad_norm", "_hj_conflict_ema", "_hj_conflict_peak",
            "_hj_adaptive_weight",
            "_hj_active_optimizer_steps",
        )
        state["hj_controller"] = {name: getattr(self, name) for name in names}
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        for name, value in (state or {}).get("hj_controller", {}).items():
            setattr(self, name, value)


def util_str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")
