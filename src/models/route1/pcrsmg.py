"""Player-Conditional Replicated Stochastic-Measure Gradient (PC-RSMG).

The native UNSB optimizer is a sequential stochastic game: D, E, then joint
G/F.  PC-RSMG estimates each player's expected gradient at the state at which
that player is actually optimized.  D and E can share a replica bundle because
their losses do not depend on one another's parameters.  G/F receive a fresh
bundle after both opponent steps are committed, removing the otherwise hidden
correlation between the opponent update and the generator's stochastic view.
"""

from __future__ import annotations

import torch


_VIEW_NAMES = (
    "times", "time_idx", "timestep", "real_A_noisy", "real_A_noisy2",
    "real", "realt", "fake", "fake_B", "fake_B2", "XtB", "idt_B",
    "flipped_for_equivariance",
)

EXPECTED_PLAYER_CONDITIONAL_SCHEDULE = (
    "DE_BUNDLE",
    "D_COMMIT",
    "E_COMMIT",
    "GF_BUNDLE",
    "GF_COMMIT",
)


def coupled_game_conditional_bias_example() -> dict[str, float]:
    """Exact two-point counterexample for stale cross-player randomness.

    Let the opponent update expose ``d'=-eta*xi`` for xi in {-1,+1}.  At that
    updated state the generator field is ``d'+zeta``.  Reusing ``zeta=xi`` has
    unit conditional bias; an independent symmetric zeta has zero conditional
    bias and the average of two independent replicas has half the conditional
    variance of one.  Enumeration is exact and contains no trained target.
    """
    eta = 0.25
    support = (-1.0, 1.0)
    stale_max_bias = 0.0
    fresh_max_bias = 0.0
    fresh_single_variances = []
    fresh_pair_variances = []
    for xi in support:
        updated_opponent = -eta * xi
        stale_mean = updated_opponent + xi
        fresh_values = [updated_opponent + zeta for zeta in support]
        pair_values = [
            updated_opponent + 0.5 * (first + second)
            for first in support for second in support
        ]
        fresh_mean = sum(fresh_values) / len(fresh_values)
        pair_mean = sum(pair_values) / len(pair_values)
        stale_max_bias = max(stale_max_bias, abs(stale_mean - updated_opponent))
        fresh_max_bias = max(
            fresh_max_bias,
            abs(fresh_mean - updated_opponent),
            abs(pair_mean - updated_opponent),
        )
        fresh_single_variances.append(
            sum((value - fresh_mean) ** 2 for value in fresh_values)
            / len(fresh_values)
        )
        fresh_pair_variances.append(
            sum((value - pair_mean) ** 2 for value in pair_values)
            / len(pair_values)
        )
    single_variance = sum(fresh_single_variances) / len(fresh_single_variances)
    pair_variance = sum(fresh_pair_variances) / len(fresh_pair_variances)
    return {
        "stale_conditional_bias_max": stale_max_bias,
        "fresh_conditional_bias_max": fresh_max_bias,
        "fresh_pair_to_single_variance_ratio": pair_variance / single_variance,
    }


class PCRSMGMixin:
    def _pcrsmg_replicates(self) -> int:
        return int(getattr(self.opt, "pcrsmg_replicates", 2))

    def _capture_pcrsmg_view(self) -> dict:
        return {name: getattr(self, name) for name in _VIEW_NAMES if hasattr(self, name)}

    def _restore_pcrsmg_view(self, view: dict) -> None:
        for name, value in view.items():
            setattr(self, name, value)

    @staticmethod
    def _mean_loss_records(records: list[dict[str, torch.Tensor | float]]) -> dict:
        result = {}
        for name in sorted({key for record in records for key in record}):
            values = [record[name] for record in records if name in record]
            tensors = [
                value.detach() if torch.is_tensor(value) else torch.tensor(float(value))
                for value in values
            ]
            device = next(
                (value.device for value in tensors if value.device.type != "cpu"), None
            )
            if device is not None:
                tensors = [value.to(device) for value in tensors]
            result[name] = sum(tensors) / float(len(tensors))
        return result

    def _pcrsmg_views(self, count: int, *, player_bundle: str) -> list[dict]:
        if player_bundle not in ("DE", "GF"):
            raise ValueError(f"unknown PC-RSMG player bundle: {player_bundle}")
        views = []
        for _ in range(count):
            self.forward()
            views.append(self._capture_pcrsmg_view())
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        self._pcrsmg_bundle_serial += 1
        if player_bundle == "DE":
            self._pcrsmg_de_bundle_count += 1
        else:
            self._pcrsmg_gf_bundle_count += 1
        self._pcrsmg_last_schedule.append(f"{player_bundle}_BUNDLE")
        return views

    def _pcrsmg_commit_event(self, event: str) -> None:
        self._pcrsmg_last_schedule.append(event)

    def optimize_parameters(self):
        replicas = self._pcrsmg_replicates()
        if replicas == 1:
            return super().optimize_parameters()
        if replicas != 2:
            raise ValueError("the frozen PC-RSMG derivation requires exactly two replicas")

        self._pcrsmg_last_schedule = []

        # D and E have independent parameter blocks and neither loss reads the
        # other block, so the same pre-opponent iid bundle is valid for both.
        de_views = self._pcrsmg_views(replicas, player_bundle="DE")

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        d_records = []
        for view in de_views:
            self._restore_pcrsmg_view(view)
            loss = self.compute_D_loss()
            d_records.append({
                "loss_D": loss,
                "loss_D_real": self.loss_D_real,
                "loss_D_fake": self.loss_D_fake,
            })
            (loss / replicas).backward()
        self.optimizer_D.step()
        self._pcrsmg_commit_event("D_COMMIT")

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        e_records = []
        for view in de_views:
            self._restore_pcrsmg_view(view)
            loss = self.compute_E_loss()
            e_records.append({"loss_E": loss})
            (loss / replicas).backward()
        self.optimizer_E.step()
        self._pcrsmg_commit_event("E_COMMIT")

        d_means = self._mean_loss_records(d_records)
        e_means = self._mean_loss_records(e_records)

        # Drop every pre-opponent graph before creating the generator bundle.
        # The new draws are therefore generated by later RNG calls and are
        # conditionally independent of the D/E estimator at the updated state.
        del de_views, d_records, e_records
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        gf_views = self._pcrsmg_views(replicas, player_bundle="GF")

        self.optimizer_G.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        g_records = []
        for view in gf_views:
            self._restore_pcrsmg_view(view)
            loss = self.compute_G_loss()
            record = {
                "loss_G": loss,
                "loss_G_GAN": self.loss_G_GAN,
                "loss_SB": self.loss_SB,
                "loss_NCE": self.loss_NCE,
            }
            if hasattr(self, "loss_NCE_Y"):
                record["loss_NCE_Y"] = self.loss_NCE_Y
            g_records.append(record)
            (loss / replicas).backward()
        self._before_generator_optimizer_step()
        self.optimizer_G.step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()
        self._pcrsmg_commit_event("GF_COMMIT")

        if tuple(self._pcrsmg_last_schedule) != EXPECTED_PLAYER_CONDITIONAL_SCHEDULE:
            raise RuntimeError("PC-RSMG player-conditional execution order changed")
        self._restore_pcrsmg_view(gf_views[-1])
        g_means = self._mean_loss_records(g_records)
        for means in (d_means, e_means, g_means):
            for name, value in means.items():
                setattr(self, name, value)
        self._pcrsmg_update_index += 1

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if self._pcrsmg_replicates() == 1:
            return state
        state["pcrsmg"] = {
            "replicates": 2,
            "update_index": int(self._pcrsmg_update_index),
            "bundle_serial": int(self._pcrsmg_bundle_serial),
            "de_bundle_count": int(self._pcrsmg_de_bundle_count),
            "gf_bundle_count": int(self._pcrsmg_gf_bundle_count),
            "last_schedule": list(self._pcrsmg_last_schedule),
        }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if self._pcrsmg_replicates() == 1:
            return
        saved = (state or {}).get("pcrsmg")
        if saved is None:
            self._initialize_pcrsmg_state()
            return
        if int(saved.get("replicates", -1)) != 2:
            raise RuntimeError("PC-RSMG checkpoint replica count mismatch")
        self._pcrsmg_update_index = int(saved["update_index"])
        self._pcrsmg_bundle_serial = int(saved["bundle_serial"])
        self._pcrsmg_de_bundle_count = int(saved["de_bundle_count"])
        self._pcrsmg_gf_bundle_count = int(saved["gf_bundle_count"])
        self._pcrsmg_last_schedule = list(saved["last_schedule"])
        if self._pcrsmg_last_schedule and (
            tuple(self._pcrsmg_last_schedule) != EXPECTED_PLAYER_CONDITIONAL_SCHEDULE
        ):
            raise RuntimeError("PC-RSMG checkpoint schedule mismatch")

    def _initialize_pcrsmg_state(self) -> None:
        self._pcrsmg_update_index = 0
        self._pcrsmg_bundle_serial = 0
        self._pcrsmg_de_bundle_count = 0
        self._pcrsmg_gf_bundle_count = 0
        self._pcrsmg_last_schedule = []
