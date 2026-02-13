from typing import Any, cast

import torch

from tianshou.algorithm.modelfree.a2c import A2CTrainingStats
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.data import SequenceSummaryStats
from tianshou.data.types import LogpOldProtocol


class GRPO(PPO):
    """Group-Relative PPO.

    If a group id is present in `batch.info[group_batch_key]`, this algorithm
    normalizes advantages within each group before policy updates.
    """

    def __init__(
        self,
        *args,
        group_advantage_normalization: bool = True,
        group_batch_key: str = "group_id",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.group_advantage_normalization = group_advantage_normalization
        self.group_batch_key = group_batch_key

    def _extract_group_ids(self, minibatch: LogpOldProtocol) -> torch.Tensor | None:
        info = getattr(minibatch, "info", None)
        if info is None:
            return None
        group_ids: Any = None
        if isinstance(info, dict):
            group_ids = info.get(self.group_batch_key)
        else:
            if hasattr(info, self.group_batch_key):
                group_ids = getattr(info, self.group_batch_key)
            elif hasattr(info, "get"):
                group_ids = info.get(self.group_batch_key, None)
        if group_ids is None:
            return None
        group_ids_t = torch.as_tensor(group_ids, device=minibatch.adv.device)
        return group_ids_t.flatten()

    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        group_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if group_ids is None:
            if not self.advantage_normalization:
                return advantages
            mean, std = advantages.mean(), advantages.std()
            return (advantages - mean) / (std + self._eps)

        flat_adv = advantages.flatten()
        flat_ids = group_ids.to(flat_adv.device).flatten()
        if flat_ids.shape[0] != flat_adv.shape[0]:
            if self.advantage_normalization:
                mean, std = advantages.mean(), advantages.std()
                return (advantages - mean) / (std + self._eps)
            return advantages

        normalized = flat_adv.clone()
        for group_id in torch.unique(flat_ids):
            mask = flat_ids == group_id
            group_adv = flat_adv[mask]
            group_mean = group_adv.mean()
            group_std = group_adv.std(unbiased=False)
            normalized[mask] = (group_adv - group_mean) / (group_std + self._eps)
        return normalized.view_as(advantages)

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
                dist = self.policy(minibatch).dist
                group_ids = (
                    self._extract_group_ids(minibatch)
                    if self.group_advantage_normalization
                    else None
                )
                advantages = self._normalize_advantages(minibatch.adv, group_ids)
                ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                value = self.critic(minibatch.obs).flatten()
                if self.value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -self.eps_clip,
                        self.eps_clip,
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
