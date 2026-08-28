"""Drop-in subclass of the clean UNSB ``SBModel`` that adds DT-CovMatch.

The algorithm is entirely contained in :class:`DTCovMatch`; this subclass only
owns the few plumbing decisions required by the CUT/UNSB training loop:

* exposing ``dtcov_*`` command-line options,
* constructing the regularizer after ``SBModel`` has created ``netG``,
* adding one scalar to ``compute_G_loss`` when the regularizer is enabled.

The evaluation path is unchanged: if ``dtcov_lambda == 0`` (the default) or the
warmup has not been reached, ``compute_G_loss`` is byte-for-byte the base model's
loss.
"""

from __future__ import annotations

import copy
import torch

from .dtcovmatch import (
    DTCovMatch,
    DTCovMatchConfig,
    domain_key_from_path,
    scheduled_lambda,
    time_norm_from_times,
)
from models.hj.diagnostics import EpochDiagnostics
from models.sb_model import SBModel


class SBModelDTCovMatch(SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)

        parser.add_argument("--dtcov_lambda", type=float, default=0.0,
                            help="additive DT-CovMatch regularization strength")
        parser.add_argument("--dtcov_lambda_schedule", type=str, default="fixed",
                            choices=["fixed", "linear_decay", "cosine_decay",
                                     "ramp_hold_cosine_decay", "ramp_hold_linear_decay",
                                     "adaptive"],
                            help="epoch schedule for --dtcov_lambda")
        parser.add_argument("--dtcov_ramp_start_epoch", type=int, default=1)
        parser.add_argument("--dtcov_ramp_end_epoch", type=int, default=0)
        parser.add_argument("--dtcov_decay_start_epoch", type=int, default=0)
        parser.add_argument("--dtcov_decay_end_epoch", type=int, default=0)
        parser.add_argument("--dtcov_lambda_min", type=float, default=0.0)
        parser.add_argument("--dtcov_adaptive_epsilon", type=float, default=0.02)
        parser.add_argument("--dtcov_adaptive_patience", type=int, default=5)
        parser.add_argument("--dtcov_adaptive_floor", type=float, default=1e-4,
                            help="drift value above which plateau tracking starts")
        parser.add_argument("--dtcov_adaptive_max_epoch", type=int, default=0,
                            help="force lambda to 0 after this epoch (0=no cap)")
        parser.add_argument("--dtcov_diag_out", type=str, default="")

        parser.add_argument("--dtcov_m", type=int, default=4,
                            help="number of stochastic endpoint proposals")
        parser.add_argument("--dtcov_region_patch", type=int, default=32)
        parser.add_argument("--dtcov_u_floor", type=float, default=1e-8)
        parser.add_argument("--dtcov_norm_eps", type=float, default=1e-4)
        parser.add_argument("--dtcov_norm_momentum", type=float, default=0.98)
        parser.add_argument("--dtcov_norm_clip", type=float, default=3.0)
        parser.add_argument("--dtcov_domain_balance", type=str, default="grouped_domain",
                            choices=["none", "equal", "grouped_domain"])
        parser.add_argument("--dtcov_teacher", type=str, default="frozen",
                            choices=["frozen", "self"],
                            help="frozen: first-use teacher snapshot; self: live generator")
        parser.add_argument("--dtcov_norm_mode", type=str, default="domain_time",
                            choices=["domain_time", "global"],
                            help="domain_time: per-(domain,time) EMA; global: single EMA")
        parser.add_argument("--dtcov_signal_norm", type=str, default="on",
                            choices=["on", "off"],
                            help="on: normalize U by directional signal; off: raw U")
        parser.add_argument("--dtcov_warmup_iters", type=int, default=300)
        parser.add_argument("--dtcov_time_mode", type=str, default="actual",
                            choices=["index", "actual"])
        parser.add_argument(
            "--dtcov_search_start_step", type=int, default=-1,
            help="search-runner override: first active optimizer step (-1=epoch schedule)",
        )
        parser.add_argument(
            "--dtcov_search_duration_steps", type=int, default=0,
            help="search-runner override: map this active window to DT ages 1..25",
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._dtcov_iter = 0
        self._dtcov_epoch = 0
        self._dtcov_epoch_loss_sum = 0.0
        self._dtcov_epoch_loss_count = 0
        self._dtcov_mismatch_ema = None
        self._dtcov_plateau = 0
        self._dtcov_activated = False
        self._dtcov_adaptive_lambda = 0.0
        self._dtcov_step_in_epoch = 0
        self._dtcov_sb_grad_norm = 0.0
        self._dtcov_diag = (
            EpochDiagnostics(getattr(opt, "dtcov_diag_out", ""))
            if getattr(opt, "dtcov_diag_out", "")
            else None
        )
        if self.isTrain:
            self.loss_names += ["U_match"]
            self.dtcov = DTCovMatch(
                netG=self.netG,
                config=self._make_dtcov_config(),
            )

    def _scheduled_dtcov_lambda(self) -> float:
        opt = self.opt
        base = float(getattr(opt, "dtcov_lambda", 0.0))
        if base <= 0.0:
            return 0.0
        search_start = int(getattr(opt, "dtcov_search_start_step", -1))
        if search_start >= 0:
            step = int(getattr(self, "_search_global_step", 0))
            duration = int(getattr(opt, "dtcov_search_duration_steps", 0))
            if duration <= 0:
                raise RuntimeError("DT search schedule requires a positive duration")
            if step < search_start or step >= search_start + duration:
                return 0.0
            active_step = step - search_start
            self._dtcov_epoch = min(25, 1 + (active_step * 25) // duration)
        if str(getattr(opt, "dtcov_lambda_schedule", "fixed")) == "adaptive":
            return self._dtcov_adaptive_lambda
        return scheduled_lambda(
            base=base,
            epoch=self._dtcov_epoch,
            schedule=str(getattr(opt, "dtcov_lambda_schedule", "fixed")),
            ramp_start=int(getattr(opt, "dtcov_ramp_start_epoch", 1)),
            ramp_end=int(getattr(opt, "dtcov_ramp_end_epoch", 0)),
            decay_start=int(getattr(opt, "dtcov_decay_start_epoch", 0)),
            decay_end=int(getattr(opt, "dtcov_decay_end_epoch", 0)),
            min_value=float(getattr(opt, "dtcov_lambda_min", 0.0)),
        )

    def _update_adaptive_lambda(self) -> None:
        base = float(getattr(self.opt, "dtcov_lambda", 0.0))
        ramp_end = int(getattr(self.opt, "dtcov_ramp_end_epoch", 0))
        patience = int(getattr(self.opt, "dtcov_adaptive_patience", 5))
        max_epoch = int(getattr(self.opt, "dtcov_adaptive_max_epoch", 0))
        if max_epoch > 0 and self._dtcov_epoch >= max_epoch:
            self._dtcov_adaptive_lambda = 0.0
        elif ramp_end > 0 and self._dtcov_epoch <= ramp_end:
            self._dtcov_adaptive_lambda = base * max(self._dtcov_epoch, 1) / ramp_end
        elif self._dtcov_plateau >= patience:
            self._dtcov_adaptive_lambda = 0.0
        else:
            self._dtcov_adaptive_lambda = base

    def _make_dtcov_config(self) -> DTCovMatchConfig:
        opt = self.opt
        return DTCovMatchConfig(
            m=int(getattr(opt, "dtcov_m", 4)),
            region_patch=int(getattr(opt, "dtcov_region_patch", 32)),
            u_floor=float(getattr(opt, "dtcov_u_floor", 1e-8)),
            norm_eps=float(getattr(opt, "dtcov_norm_eps", 1e-4)),
            norm_momentum=float(getattr(opt, "dtcov_norm_momentum", 0.98)),
            norm_clip=float(getattr(opt, "dtcov_norm_clip", 3.0)),
            domain_balance=str(getattr(opt, "dtcov_domain_balance", "grouped_domain")),
            lambda_value=float(getattr(opt, "dtcov_lambda", 0.0)),
            warmup_iters=int(getattr(opt, "dtcov_warmup_iters", 300)),
            time_mode=str(getattr(opt, "dtcov_time_mode", "actual")),
            latent_dim=4 * int(opt.ngf),
            freeze_teacher=(str(getattr(opt, "dtcov_teacher", "frozen")) == "frozen"),
            norm_mode=str(getattr(opt, "dtcov_norm_mode", "domain_time")),
            signal_normalize=(str(getattr(opt, "dtcov_signal_norm", "on")) == "on"),
        )

    def _dtcov_inputs(self):
        X_t = self.real_A_noisy.detach()
        time_idx = self.time_idx
        time_scalar = time_idx[0] if torch.is_tensor(time_idx) and time_idx.dim() > 0 else time_idx
        time_id = int(time_scalar.detach().item() if torch.is_tensor(time_scalar) else time_scalar)
        t_norm = time_norm_from_times(self.times, time_scalar, int(self.opt.num_timesteps))
        paths = list(getattr(self, "image_paths", []) or [])
        domain_keys = [
            domain_key_from_path(paths[idx] if idx < len(paths) else "unknown")
            for idx in range(int(X_t.size(0)))
        ]
        return X_t, time_idx, time_id, t_norm, domain_keys

    def compute_G_loss(self):
        base_loss = super().compute_G_loss()
        if not self.isTrain or not hasattr(self, "dtcov"):
            self.loss_U_match = torch.zeros_like(base_loss).detach()
            return base_loss

        self._dtcov_step_in_epoch += 1
        if self._dtcov_step_in_epoch == 1 and self._dtcov_diag is not None:
            self._dtcov_sb_grad_norm = self._compute_sb_entropy_grad_norm()

        self.dtcov.config.lambda_value = self._scheduled_dtcov_lambda()
        X_t, time_idx, time_id, t_norm, domain_keys = self._dtcov_inputs()
        reg, diag = self.dtcov.forward(
            X_t=X_t,
            time_idx=time_idx,
            time_id=time_id,
            t_norm=t_norm,
            domain_keys=domain_keys,
        )
        self.loss_U_match = reg.detach()
        self._dtcov_epoch_loss_sum += float(self.loss_U_match.detach().item())
        self._dtcov_epoch_loss_count += 1
        self.loss_G = base_loss + self.dtcov.config.lambda_value * reg
        return self.loss_G

    def _compute_sb_entropy_grad_norm(self) -> float:
        loss = getattr(self, "loss_SB", None)
        if not torch.is_tensor(loss):
            return 0.0
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
            return total ** 0.5
        except Exception:
            return 0.0

    def optimize_parameters(self):
        super().optimize_parameters()
        self._dtcov_iter += 1
        if hasattr(self, "dtcov"):
            self.dtcov.iter = self._dtcov_iter

    def update_learning_rate(self):
        super().update_learning_rate()
        self._dtcov_epoch += 1
        if self._dtcov_epoch_loss_count > 0:
            epoch_mean = self._dtcov_epoch_loss_sum / self._dtcov_epoch_loss_count
            floor = float(getattr(self.opt, "dtcov_adaptive_floor", 1e-4))
            if self._dtcov_mismatch_ema is None:
                self._dtcov_mismatch_ema = epoch_mean
            else:
                prev = self._dtcov_mismatch_ema
                self._dtcov_mismatch_ema = 0.9 * prev + 0.1 * epoch_mean
                if epoch_mean > floor:
                    self._dtcov_activated = True
                    rel = abs(self._dtcov_mismatch_ema - prev) / max(abs(prev), 1e-9)
                    eps = float(getattr(self.opt, "dtcov_adaptive_epsilon", 0.02))
                    if rel < eps:
                        self._dtcov_plateau += 1
                    else:
                        self._dtcov_plateau = 0
                else:
                    self._dtcov_plateau = 0
        self._dtcov_epoch_loss_sum = 0.0
        self._dtcov_epoch_loss_count = 0
        if str(getattr(self.opt, "dtcov_lambda_schedule", "fixed")) == "adaptive":
            self._update_adaptive_lambda()
        if self._dtcov_diag is not None:
            self._dtcov_diag.log(
                epoch=self._dtcov_epoch,
                drift=self._dtcov_mismatch_ema,
                plateau=self._dtcov_plateau,
                lambda_value=self._dtcov_adaptive_lambda,
                sb_entropy_grad_norm=self._dtcov_sb_grad_norm,
            )
        self._dtcov_step_in_epoch = 0

    def set_train_epoch(self, epoch):
        """Map physical epoch to DT active age (physical e21 -> age 1)."""
        self._dtcov_epoch = max(0, int(epoch) - 20)

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        controller_names = (
            "_dtcov_iter", "_dtcov_epoch", "_dtcov_epoch_loss_sum",
            "_dtcov_epoch_loss_count", "_dtcov_mismatch_ema",
            "_dtcov_plateau", "_dtcov_activated",
            "_dtcov_adaptive_lambda", "_dtcov_step_in_epoch",
            "_dtcov_sb_grad_norm",
        )
        state["dtcov_controller"] = {
            name: getattr(self, name) for name in controller_names
        }
        if hasattr(self, "dtcov"):
            teacher = getattr(self.dtcov, "teacher", None)
            state["dtcov_core"] = {
                "iter": int(self.dtcov.iter),
                "stats_store": copy.deepcopy(self.dtcov.stats.store),
                "teacher": None if teacher is None else {
                    key: value.detach().cpu()
                    for key, value in teacher.state_dict().items()
                },
            }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        state = state or {}
        for name, value in state.get("dtcov_controller", {}).items():
            setattr(self, name, value)
        core = state.get("dtcov_core")
        if core is not None and hasattr(self, "dtcov"):
            self.dtcov.iter = int(core.get("iter", 0))
            self.dtcov.stats.store = copy.deepcopy(core.get("stats_store", {}))
            teacher_state = core.get("teacher")
            if teacher_state is not None:
                self.dtcov.inject_teacher(teacher_state)
