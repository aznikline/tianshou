from typing import cast

import numpy as np
import torch

from tianshou.algorithm.modelfree.a2c import A2CTrainingStats
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.data import SequenceSummaryStats
from tianshou.data.types import LogpOldProtocol


class DAPO(PPO):
    """Dynamic-Adaptive PPO.

    This variant keeps PPO's objective and adaptively adjusts clip range based on
    an approximate KL target, which can stabilize updates across training phases.
    """

    def __init__(
        self,
        *args,
        target_kl: float | None = 0.02,
        clip_adaptation_rate: float = 0.1,
        min_eps_clip: float = 0.05,
        max_eps_clip: float = 0.4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if clip_adaptation_rate <= 0.0 or clip_adaptation_rate >= 1.0:
            raise ValueError(
                f"`clip_adaptation_rate` must be in (0, 1), got {clip_adaptation_rate}.",
            )
        if min_eps_clip <= 0.0:
            raise ValueError(f"`min_eps_clip` must be > 0, got {min_eps_clip}.")
        if max_eps_clip < min_eps_clip:
            raise ValueError(
                f"`max_eps_clip` must be >= `min_eps_clip`, got {max_eps_clip} < {min_eps_clip}.",
            )
        if target_kl is not None and target_kl <= 0.0:
            raise ValueError(f"`target_kl` must be > 0 when provided, got {target_kl}.")

        self.target_kl = target_kl
        self.clip_adaptation_rate = clip_adaptation_rate
        self.min_eps_clip = min_eps_clip
        self.max_eps_clip = max_eps_clip
        self.current_eps_clip = float(self.eps_clip)

    def _adapt_clip_range(self, approx_kl: torch.Tensor) -> None:
        if self.target_kl is None:
            return
        kl_value = abs(float(approx_kl.item()))
        high_kl = self.target_kl * 1.5
        low_kl = self.target_kl / 1.5
        if kl_value > high_kl:
            self.current_eps_clip *= 1.0 - self.clip_adaptation_rate
        elif kl_value < low_kl:
            self.current_eps_clip *= 1.0 + self.clip_adaptation_rate
        self.current_eps_clip = float(
            np.clip(self.current_eps_clip, self.min_eps_clip, self.max_eps_clip)
        )

    def _update_with_batch(  # type: ignore[override]
        self,
        batch: LogpOldProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> A2CTrainingStats:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or -1
        for step in range(repeat):
            if self.recompute_adv and step > 0:
                batch = cast(
                    LogpOldProtocol,
                    self._add_returns_and_advantages(batch, self._buffer, self._indices),
                )
            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1
                advantages = minibatch.adv
                dist = self.policy(minibatch).dist
                if self.advantage_normalization:
                    mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (std + self._eps)

                log_prob = dist.log_prob(minibatch.act)
                approx_kl = (minibatch.logp_old - log_prob).mean()
                self._adapt_clip_range(approx_kl)
                current_eps = self.current_eps_clip

                ratios = (log_prob - minibatch.logp_old).exp().float()
                ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1.0 - current_eps, 1.0 + current_eps) * advantages
                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                value = self.critic(minibatch.obs).flatten()
                if self.value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -current_eps,
                        current_eps,
                    )
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()

                ent_loss = dist.entropy().mean()
                loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
                self.optim.step(loss)
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return A2CTrainingStats(
            loss=SequenceSummaryStats.from_sequence(losses),
            actor_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
            gradient_steps=gradient_steps,
        )
