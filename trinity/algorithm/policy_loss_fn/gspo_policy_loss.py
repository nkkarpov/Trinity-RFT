"""GSPO-token policy loss function.

Implemented from https://arxiv.org/pdf/2507.18071
"""

from typing import Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import masked_mean


@POLICY_LOSS_FN.register_module("gspo")
class GSPOLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        clip_range: Optional[float] = None,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
    ) -> None:
        super().__init__(backend=backend)
        if clip_range_low is None:
            if clip_range is None:
                raise ValueError("Either clip_range or clip_range_low must be specified.")
            self.clip_range_low = clip_range
        else:
            self.clip_range_low = clip_range_low

        if clip_range_high is None:
            if clip_range is None:
                raise ValueError("Either clip_range or clip_range_high must be specified.")
            self.clip_range_high = clip_range
        else:
            self.clip_range_high = clip_range_high

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        negative_approx_kl = logprob - old_logprob
        seq_lengths = torch.sum(action_mask, dim=-1).clamp(min=1).unsqueeze(-1)
        negative_approx_kl = negative_approx_kl / seq_lengths
        log_seq_importance_ratio = logprob - logprob.detach() + negative_approx_kl.detach()
        ratio = torch.exp(log_seq_importance_ratio)
        ppo_kl = masked_mean(-negative_approx_kl, action_mask)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_range_low, 1.0 + self.clip_range_high
        )

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), action_mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), action_mask)
        metrics = {
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_loss": pg_loss.detach().item(),
        }
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "clip_range": 0.2,
        }
