import torch
import tqdm

from collections import namedtuple, defaultdict

from fairseq.logging.meters import AverageMeter
from maskgan.runs.distributed_model import MaskGANModel
from maskgan.utils.logger import get_tqdm_config
from maskgan.utils.sequence_recovery import pretty_print
from maskgan.utils.saver import Saver
from maskgan.utils.summary import TensorboardSummary
from dataloaders import make_data_loader


class MaskGANTrainer(object):
    def __init__(self, args, check_path):

        self.pretrain = False

        # Define Dataloader
        kwargs = {'pin_memory': False, 'num_workers': 8}
        self.train_loader, self.val_loader, \
        self.test_loader, self.vocab = make_data_loader(args, **kwargs)

        # Define task
        Task = namedtuple('Task', 'source_dictionary target_dictionary')
        task = Task(source_dictionary=self.vocab,
                    target_dictionary=self.vocab)

        # Define saver
        self.saver = Saver(check_path)

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.path)
        self.writer = self.summary.create_summary()

        # Define maskgan model
        model = MaskGANModel(args, task, pretrain=self.pretrain)

        # Define maskgan optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Define learning scheduler
        self.lr_scheduler = \
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

        # Using cuda
        self.model, self.optimizer = model, optimizer
        self.model = self.model.to(device=args.cuda)

        # Define saver
        self.saver.load('maskgan', self.model)

        self.step = 0
        self.critic_lag_max = 50
        self.critic_lag = self.critic_lag_max

        self.args = args
        self.task = task

    def run(self, epochs):

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='blue')) as pbar:
            
            train_best_G_loss = float('-inf')
            best_G_loss = float('-inf')  # for reward plot 
            best_epoch = 0

            for epoch in range(1, epochs + 1):

                # Train & Validation
                train_history = self.train(self.train_loader, current_epoch=epoch)
                valid_history = self.validation(self.val_loader, current_epoch=epoch)

                # Epoch history (loss)
                epoch_history = {
                    'D_real_loss': {
                        'train': train_history.get('D_real_loss'),
                        'valid': valid_history.get('D_real_loss'),
                    },
                    'D_fake_loss': {
                        'train': train_history.get('D_fake_loss'),
                        'valid': valid_history.get('D_fake_loss'),
                    },
                    'C_loss': {
                        'train': train_history.get('C_loss'),
                        'valid': valid_history.get('C_loss'),
                    },
                    'G_loss': {
                        'train': train_history.get('G_loss'),
                        'valid': valid_history.get('G_loss'),
                    }
                }

                # Tensorboard summary
                for metric_name, metric_dict in epoch_history.items():
                    self.writer.add_scalars(
                        main_tag=metric_name,
                        tag_scalar_dict=metric_dict,
                        global_step=epoch
                    )

                # Save model if it is the current best
                valid_G_loss = epoch_history['G_loss']['valid']
                if valid_G_loss > best_G_loss:
                    best_G_loss = valid_G_loss
                    best_epoch = epoch
                    self.saver.checkpoint(f'{best_epoch}_val_maskgan', self.model)

                # Save model if it is the current best (train)
                train_G_loss = epoch_history['G_loss']['train']
                if train_G_loss > train_best_G_loss:
                    train_best_G_loss = train_G_loss
                    best_epoch = epoch
                    self.saver.checkpoint(f'{best_epoch}_train_maskgan', self.model)
                
                # Save every epoch
                self.saver.checkpoint(f'{epoch}_maskgan', self.model)

                # Logging
                desc = f" Epoch [{epoch:>04}/{epochs:>04} |"
                for metric_name, metric_dict in epoch_history.items():

                    for k, v in metric_dict.items():
                        desc += f" {k}_{metric_name}: {v:.4f} |"

                pbar.set_description_str(desc)
                pbar.update(1)

        return epoch_history

    def train(self, train_data_loader, current_epoch: int):
        steps_per_epoch = len(train_data_loader)
        self.model.train()

        num_rollouts = 1 if self.pretrain else self.args.num_rollouts  # Average loss using rollouts

        d_real_losses, d_fake_losses, c_losses, g_losses = 0.0, 0.0, 0.0, 0.0
        with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='green')) as pbar:
            for i, samples in enumerate(self.train_loader):

                samples = [sample.to(device=self.args.cuda) for sample in samples]

                # Discriminator
                real_loss, fake_loss = self.rollout_discriminator(num_rollouts, samples)
                d_real_losses += real_loss
                d_fake_losses += fake_loss

                # Generator
                g_loss = self.rollout_generator(num_rollouts, samples)
                g_losses += g_loss

                # Critic
                c_loss = self.rollout_critic(num_rollouts, samples)
                c_losses += c_loss

                desc = f" Batch [{i+1:>04}/{len(self.train_loader):>04}"
                pbar.set_description_str(desc)
                pbar.update(1)

        self.lr_scheduler.step(current_epoch)

        d_real_losses /= (i + 1)
        d_fake_losses /= (i + 1)
        c_losses /= (i + 1)
        g_losses /= (i + 1)

        total_loss_out = {
            'D_real_loss': d_real_losses,
            'D_fake_loss': d_fake_losses,
            'C_loss': c_losses,
            'G_loss': g_losses}

        self.step += 1

        return total_loss_out

    def rollout_discriminator(self, num_rollouts, samples):
        srcs, tgts, lengths, mask = samples

        real, fake = AverageMeter(), AverageMeter()

        batch_size, seq_len = samples[0].size()

        self.optimizer.zero_grad()

        for rollout in range(num_rollouts):
            real_loss = self.model(
                srcs, lengths, mask, tgts,
                tag='d-step', real=True)

            real_loss = real_loss.sum() / batch_size

            with torch.no_grad():
                net_output = self.model(
                    srcs, lengths, mask, tgts,
                    tag='g-step')

                generated = net_output[1]

            fake_loss = self.model(
                srcs, lengths, mask, generated,
                tag='d-step', real=False)

            fake_loss = fake_loss.sum() / batch_size

            loss = (real_loss + fake_loss) / 2
            loss.backward()

            real.update(real_loss.item())
            fake.update(fake_loss.item())

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optim_clip_value)  # add gradient clip
        self.optimizer.step()

        return real.avg, fake.avg

    def rollout_critic(self, num_rollouts, samples):
        srcs, tgts, lengths, mask = samples

        batch_size, seq_len = samples[0].size()
        meter = AverageMeter()

        self.optimizer.zero_grad()

        for rollout in range(num_rollouts):
            loss = self.model(srcs, lengths, mask, tgts, tag='c-step')
            loss = loss.sum() / batch_size
            loss.backward()

            meter.update(loss.item())

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optim_clip_value)  # add gradient clip
        self.optimizer.step()

        return meter.avg

    def rollout_generator(self, num_rollouts, samples):
        srcs, tgts, lengths, mask = samples

        batch_size, seq_len = samples[0].size()
        meter = AverageMeter()

        ppl_meter = defaultdict(lambda: AverageMeter())

        self.optimizer.zero_grad()

        for rollout in range(num_rollouts):
            loss, generated, ppl = self.model(srcs, lengths, mask, tgts, tag='g-step')
            loss = loss.sum() / batch_size
            loss.backward()

            meter.update(-1 * loss.item())
            # meter.update(loss.item())

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optim_clip_value)  # add gradient clip
        self.optimizer.step()
        self.debug('train', samples, generated)

        return meter.avg

    def debug(self, key, samples, generated):
        srcs, tgts, lengths, mask = samples
        tag = f'generated/{key}'
        # logger = lambda s: self.logger.log(tag, s)

        pretty_print(self.vocab, srcs, tgts, generated, truncate=10)

    def validation(self, val_data_loader, current_epoch):
        self.model.eval()

        _meters = 'generator dfake dreal critic ppl_sampled ppl_truths'
        _n_meters = len(_meters.split())

        Meters = namedtuple('Meters', _meters)
        meters_list = [AverageMeter() for i in range(_n_meters)]
        meters = Meters(*meters_list)
        
        steps_per_epoch = len(val_data_loader)
        with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='yellow')) as pbar:
            for i, samples in enumerate(val_data_loader):
                samples = [sample.to(self.args.cuda) for sample in samples]
                self._validate(current_epoch, meters, samples)

                desc = f" Batch [{i+1:>04}/{len(self.val_loader):>04}"
                pbar.set_description_str(desc)
                pbar.update(1)

        total_loss_out = {
            'D_real_loss': meters.dreal.avg,
            'D_fake_loss': meters.dfake.avg,
            'C_loss': meters.critic.avg,
            'G_loss': meters.generator.avg}

        return total_loss_out

    def aggregate(self, batch_size):

        return lambda tensor: tensor.sum() / batch_size

    def _validate(self, epoch, meters, samples):

        with torch.no_grad():
            srcs, tgts, lengths, mask = samples

            batch_size, seq_len = samples[0].size()

            agg = self.aggregate(batch_size)

            real_loss = self.model(
                srcs, lengths, mask, tgts,
                tag='d-step', real=True)

            real_loss = agg(real_loss)

            generator_loss, generated, ppl = self.model(
                srcs, lengths, mask, tgts,
                tag='g-step', ppl=True)

            generator_loss = agg(-1 * generator_loss)

            fake_loss = self.model(
                srcs, lengths, mask, generated,
                tag='d-step', real=False)

            fake_loss = agg(fake_loss)

            loss = (real_loss + fake_loss) / 2

            critic_loss = self.model(
                srcs, lengths, mask, tgts,
                tag='c-step')

            critic_loss = agg(critic_loss)

            meters.dreal.update(real_loss.item())
            meters.dfake.update(fake_loss.item())
            meters.generator.update(generator_loss.item())
            meters.critic.update(critic_loss.item())

            self.debug('dev', samples, generated)

            for key in ppl:
                ppl[key] = agg(ppl[key])

            meters.ppl_sampled.update(ppl['sampled'].item())
            meters.ppl_truths.update(ppl['ground-truth'].item())

            self.debug('dev', samples, generated)
