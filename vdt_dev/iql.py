from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.iql import ImplicitQLearning as BaseImplicitQLearning
from src.iql import asymmetric_l2_loss
from src.util import sample_batch, update_exponential_moving_average
from vdt_dev.value_conditioning import compute_detached_routing_values


class DevImplicitQLearning(BaseImplicitQLearning):
    """
    Dev-only IQL wrapper that preserves the original losses while adding Step 2 routing
    context plumbing for detached value-conditioned queries.
    """

    def compute_detached_routing_values(
        self,
        states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return compute_detached_routing_values(self.policy, self.vf, states)

    def update(
        self,
        dataset: Dict[str, torch.Tensor],
        batch_size: int,
        DT_batch: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> Dict[str, float]:
        if DT_batch is None:
            raise ValueError("DT_batch is required for the dev actor update.")

        self.qf.train()
        self.vf.train()
        self.policy.train()
        self.q_target.eval()

        logs: Dict[str, float] = {}

        for _ in range(100):
            batch = sample_batch(dataset, batch_size)
            observations = batch["observations"]
            actions = batch["actions"]
            next_observations = batch["next_observations"]
            rewards = batch["rewards"]
            terminals = batch["terminals"]

            with torch.no_grad():
                target_q = self.q_target(observations, actions)
                next_v = self.vf(next_observations)

            adv = target_q - self.vf(observations)
            v_loss = asymmetric_l2_loss(adv, self.tau)
            self.v_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            self.v_optimizer.step()

            targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
            targets = targets.detach()
            qs = self.qf.both(observations, actions)
            qf1_loss = F.mse_loss(qs[0], targets)
            qf2_loss = F.mse_loss(qs[1], targets)

            self.qf1_optimizer.zero_grad(set_to_none=True)
            qf1_loss.backward()
            self.qf1_optimizer.step()
            self.qf2_optimizer.zero_grad(set_to_none=True)
            qf2_loss.backward()
            self.qf2_optimizer.step()

            update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        (
            states,
            actions,
            rewards,
            action_target,
            _dones,
            rtg,
            timesteps,
            attention_mask,
            _next_observations,
            _terminals,
        ) = DT_batch

        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        routing_values = self.compute_detached_routing_values(states)

        state_preds, action_preds, _reward_preds = self.policy.forward(
            states,
            actions,
            rewards,
            action_target,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
            routing_values=routing_values,
        )

        action_preds_flat = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        action_target_flat = action_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        state_preds = state_preds[:, :-1]
        state_target = states[:, 1:]
        states_loss = ((state_preds - state_target) ** 2)[attention_mask[:, :-1] > 0].mean()

        bc_losses = F.mse_loss(action_preds_flat, action_target_flat) + states_loss
        policy_loss = bc_losses

        actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        qs = self.qf.both(actor_states, action_preds_flat)
        if np.random.uniform() > 0.5:
            q_loss = -qs[0].mean() / qs[1].abs().mean().detach()
        else:
            q_loss = -qs[1].mean() / qs[0].abs().mean().detach()

        policy_loss = policy_loss + q_loss

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        logs["BC Loss"] = float(bc_losses.item())
        logs["Actor Loss"] = float(policy_loss.item())
        logs["QL Loss"] = float(q_loss.item())
        logs["Q Loss"] = float(q_loss.item())
        logs["Value Loss"] = float(v_loss.item())
        logs["QF1 Loss"] = float(qf1_loss.item())
        logs["QF2 Loss"] = float(qf2_loss.item())
        return logs
