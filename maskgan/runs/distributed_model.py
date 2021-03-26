import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

from maskgan.utils.perplexity import perplexity, greed_sample
from maskgan.criterions import TBCELoss, WeightedMSELoss, REINFORCE, TCELoss
from maskgan.maskgan_networks import MaskGANGenerator, MaskGANDiscriminator, MaskGANCritic


class MaskGANModel(nn.Module):
    def __init__(self, args, task, pretrain=False):
        super(MaskGANModel, self).__init__()

        self.args = args

        # Define generator & cuda
        self.generator = MaskGANGenerator(
            args=args,
            task=task)

        reinforce = REINFORCE(args)
        self.gcriterion = reinforce

        # Define Discriminator & cuda
        self.discriminator = MaskGANDiscriminator(
            args=args,
            task=task)

        tbceloss = TBCELoss(args)
        self.dcriterion = tbceloss

        # Define Critic * cuda
        self.critic = MaskGANCritic(
            args=args,
            task=task)

        mse_loss = WeightedMSELoss(args)
        self.ccriterion = mse_loss

        self.pretrain = pretrain

    def forward(self, srcs, lengths, mask, tgts, **kwargs):

        if 'ppl' not in kwargs:
            kwargs['ppl'] = False

        if kwargs['tag'] == 'g-step':

            return self._gstep(srcs, lengths, mask, tgts, ppl_compute=kwargs['ppl'])

        elif kwargs['tag'] == 'c-step':

            return self._cstep(srcs, lengths, mask, tgts)

        return self._dstep(srcs, lengths, mask, tgts, real=kwargs['real'])

    def _cstep(self, srcs, lengths, mask, tgts):

        with torch.no_grad():
            samples, log_probs, attns = self.generator(srcs, lengths, tgts, mask)
            logits, attn_scores = self.discriminator(srcs, lengths, samples)  # samples == prev output token

        baselines, _ = self.critic(srcs, lengths, samples)

        with torch.no_grad():
            reward, cumulative_rewards = self.gcriterion(log_probs, logits, mask, baselines)  # Loss == reward

        critic_loss = self.ccriterion(baselines.squeeze(2), cumulative_rewards, mask)

        return critic_loss

    def _gstep(self, srcs, lengths, mask, tgts, ppl_compute=False):
        samples, log_probs, attns = self.generator(srcs, lengths, tgts, mask)

        # discriminator
        with torch.no_grad():
            logits, attn_scores = self.discriminator(srcs, lengths, samples)
            baselines, _ = self.critic(srcs, lengths, samples)

        reward, cumulative_rewards = self.gcriterion(log_probs, logits, mask, baselines.detach())
        loss = -1 * reward
        # loss = reward

        # Compute perplexity
        if ppl_compute:

            with torch.no_grad():
                logits = self.generator.logits(srcs, lengths, tgts, mask).clone()
                log_probs = F.log_softmax(logits, dim=2)
                ppl = perplexity(tgts, samples, log_probs)

        else:
            ppl = None

        return loss, samples, ppl

    def _dstep(self, srcs, lengths, mask, tgts, real=True):
        logits, attn_scores = self.discriminator(srcs, lengths, tgts)
        mask = mask.unsqueeze(2)
        truths = torch.ones_like(logits) if real else torch.ones_like(logits) - mask

        loss = self.dcriterion(logits, truths, weight=mask)

        return loss
