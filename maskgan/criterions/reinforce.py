import torch
import torch.nn as nn


class REINFORCE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.gamma = args.gamma
        self.clip_value = args.clip_value
        self.log_sigmoid = torch.nn.LogSigmoid().to(self.args.cuda)

    def forward(self, log_probs, logits, weight, baselines=None):
        batch_size, seq_len, _ = logits.size()
        rewards = self.log_sigmoid(logits).squeeze(2)

        cumulative_rewards = []
        for t in range(seq_len):
            cum_value = rewards.new_zeros(batch_size)

            for s in range(t, seq_len):
                exp = float(s - t)
                k = (self.gamma ** exp)

                cum_value += k * weight[:, s] * rewards[:, s]

            cumulative_rewards.append(cum_value)

        cumulative_rewards = torch.stack(cumulative_rewards, dim=1)

        if baselines is not None:
            baselines = baselines.squeeze(2)
            advantages = cumulative_rewards - baselines  # advantage function

        else:
            advantages = cumulative_rewards

        advantages = advantages - advantages.mean(dim=0)
        advantages = advantages.clamp(-1 * self.clip_value, self.clip_value)

        generator_objective = (advantages * log_probs).sum(dim=0)

        return generator_objective, cumulative_rewards.clone()
