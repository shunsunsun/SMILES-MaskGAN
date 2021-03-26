import torch
import torch.nn as nn


class TBCELoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(self.args.cuda)

    def forward(self, pred_logits, truths, weight=None):

        B, T, H = pred_logits.size()
        truths = truths.float()
        weight = weight.float()

        loss = self.criterion(pred_logits, truths)

        return loss.sum()


class TCELoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.args.cuda)

    def forward(self, logits, truths, weight=None):
        logits = logits.contiguous()

        B, T, H = logits.size()
        logits = logits.view(T * B, H)
        target = truths.contiguous().view(-1)

        loss = self.criterion(logits, target)

        return loss


class WeightedMSELoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.criterion = nn.MSELoss(reduction='none').to(self.args.cuda)

    def forward(self, preds, truths, weights):
        mse_loss = self.criterion(preds, truths)

        return mse_loss
        # return weights * mse_loss / weights.sum()


def _debug(pred_logits, truths, weight):
    B, T, H = pred_logits.size()

    for b in range(B):
        npreds = pred_logits[b, :, :].view(-1)
        ntruths = truths[b, :, :].view(-1)
        nweights = weight[b, :].view(-1)

        weighted = nn.BCEWithLogitsLoss(reduction='none')(npreds, ntruths) * nweights

        outstr = """
        sizes: {} {} {}
        predns:  {}
        truths:  {}
        weights: {}
        final:   {}
                """.format(npreds.size(), ntruths.size(), nweights.size(),
                           torch.sigmoid(npreds).tolist(),
                           ntruths.tolist(),
                           nweights.tolist(),
                           weighted.tolist()
                           )
        print(outstr, flush=True)

        break
