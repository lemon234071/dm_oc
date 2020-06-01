import os
import sys
import time
import math
import logging
import datetime
import platform
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from tensorboardX import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, CONFIG_NAME, AdamW
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from data_utils.inputter import build_dataloader
from data_utils.data_process import build_from_json, build_one_json
from weighting.optim import warmup_linear, noam_decay, noamwd_decay, exponential_decay
from weighting.decode_utils import *
from weighting.eval_utils import *

from magic_module import MagicModule

logger = logging.getLogger(__file__)

# MAX_SEQ_LENGTH = 64
EPSILON = 1e-5
SYS = platform.system()

class LMManager:
    def __init__(self, args, pretrained, model_checkpoint, report_every,
                 ren, norm_fn, device, logdir=None):
        self.args = args
        self._ren = ren
        self._device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            model_checkpoint, do_lower_case=True)
        self.pad_id = self.tokenizer.eos_token_id

        self._config = GPT2Config.from_json_file(os.path.join(model_checkpoint, CONFIG_NAME))
        self._max_len = 256  # 512  # self._config.n_ctx

        self._model = GPT2LMHeadModel.from_pretrained(
            model_checkpoint).to(device) if pretrained else GPT2LMHeadModel(self._config).to(device)
        num_param, _, __ = _tally_parameters(self._model)
        logger.info("model paramerters: {}".format(num_param))

        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        self.save_dir = os.path.join("checkpoints", args.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        elif args.infer_from == "":
            if SYS != "Windows":
                raise Exception("path exists {}".format(self.save_dir))

        self._optimizer = None
        self.writer = SummaryWriter(logdir=logdir)
        self.report_every = report_every
        self.batch_step = 0
        self.training_step = 1
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_val_step = args.max_val_step

        self._dataset = {}
        self._data_loader = {}

        self._weights = None
        self._w_decay = None

        if norm_fn == 'linear':
            self._norm_fn = _linear_normalize
        elif norm_fn == 'softmax':
            self._norm_fn = _softmax_normalize

        if ren:
            assert norm_fn == 'linear'

    def init_weights(self, w_init, w_decay):
        if self._ren:
            raise ValueError(
                'no global weighting initialization when \'ren\'=True')

        self._weights = torch.tensor(
            [w_init] * len(self._dataset['train']['all_ids']), requires_grad=True).to(device=self._device)
        self._w_decay = w_decay

    def load_data(self, set_type, examples, batch_size, shuffle):
        self._dataset[set_type] = build_from_json(examples, self.tokenizer, self._max_len)
        self._data_loader[set_type] = build_dataloader(
            self._dataset[set_type],
            batch_size=batch_size,
            shuffle=shuffle,
            pad_id=self.pad_id,
            batch_first=True)

    def get_optimizer(self, learning_rate):
        self._optimizer = _get_optimizer(
            self._model, learning_rate=learning_rate)

    def pretrain_epoch(self):
        self._model.train()

        start_t = time.time()
        for step, batch in enumerate(tqdm(self._data_loader['train'],
                                          desc='Pre-training')):
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, segment_ids, label_ids, _ = batch

            (loss), *_ = self._model(input_ids, token_type_ids=segment_ids, labels=label_ids)
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.args.max_norm)
            self.batch_step += 1
            if self.batch_step % self.gradient_accumulation_steps == 0:
                _set_lr(self._optimizer, self.training_step,
                        self.args.lr_schedule, self.args.learning_rate,
                        self.args.warmup_steps, self.args.warmup_proportion,
                        self._config.n_embd, self.args.num_optim_steps)

                self._optimizer.step()
                self._optimizer.zero_grad()
                self.training_step += 1
                if self.training_step % self.report_every == 0:
                    self._stat(loss * self.gradient_accumulation_steps, start_t)

    def train_epoch(self):
        self._model.train()
        loss_accum, non_pad_accum = 0.0, 0
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        start_t = time.time()

        for step, batch in enumerate(tqdm(self._data_loader['train'],
                                          desc='Training', mininterval=1)):
            weights = self._get_weights(batch)
            weights = self._norm_fn(weights)
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, token_type_ids, label_ids, _ = batch

            logits, *_ = self._model(input_ids, token_type_ids=token_type_ids)
            loss_original, non_padding = _compute_loss(logits, label_ids, criterion, label_shape=True)
            loss = torch.sum(loss_original * weights.data.unsqueeze(1))
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.args.max_norm)
            self.batch_step += 1
            loss_accum += loss_original.sum().item()
            non_pad_accum += non_padding

            if self.batch_step % self.gradient_accumulation_steps == 0:
                _set_lr(self._optimizer, self.training_step,
                        self.args.lr_schedule, self.args.learning_rate,
                        self.args.warmup_steps, self.args.warmup_proportion,
                        self._config.n_embd, self.args.num_optim_steps)

                self._optimizer.step()
                self._optimizer.zero_grad()
                self.training_step += 1
                if self.training_step % self.report_every == 0:
                    self._stat(loss_accum / non_pad_accum, start_t)
                    loss_accum, non_pad_accum = 0, 0

    def _get_weights(self, batch):
        batch = tuple(t.to(self._device) for t in batch)
        input_ids, segment_ids, label_ids, ids = batch
        batch_size = label_ids.shape[0]

        optimizer_initialized = ('exp_avg' in self._optimizer.state[
            next(self._model.parameters())])
        if not optimizer_initialized:
            return torch.ones(batch_size).to(self._device)

        if self._ren:
            weights = torch.zeros(
                batch_size, requires_grad=True).to(self._device)
        else:
            weights = self._weights[ids]

        magic_model = MagicModule(self._model)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        for i in range(batch_size):
            # compute grads L_theta
            self._model.zero_grad()
            # TODO(yida) why here compute loss outside the model ?
            # (loss), *_ = self._model(
            #     input_ids[i:i + 1], token_type_ids=segment_ids[i:i + 1], labels=label_ids[i:i + 1])
            logits, *_ = self._model(
                input_ids[i:i + 1], token_type_ids=segment_ids[i:i + 1])
            loss, _ = _compute_loss(logits, label_ids[i:i + 1], criterion)

            grads = torch.autograd.grad(
                loss, [param for name, param in self._model.named_parameters()])
            grads = {param: grads[j] for j, (name, param) in enumerate(
                self._model.named_parameters())}

            # theta'(phi) = theta + grads
            deltas = _adam_delta(self._optimizer, self._model, grads)
            deltas = {name: weights[i] * delta.data for name, delta in
                      deltas.items()}
            magic_model.update_params(deltas)
        batch = tuple(t.cpu() for t in batch)

        # get grad of phi with dev
        weights_grad_list = []
        for step, val_batch in enumerate(self._data_loader['dev_shuffle']):
            if step > self.max_val_step:
                break

            val_batch = (t.to(self._device) for t in val_batch)
            val_input_ids, val_segment_ids, val_label_ids, _ = \
                val_batch
            val_batch_size = val_label_ids.shape[0]

            # L(theta'(phi))
            (val_loss), *_ = magic_model(
                val_input_ids, token_type_ids=val_segment_ids, labels=val_label_ids)
            val_loss = val_loss * \
                       float(val_batch_size) / float(len(self._dataset['dev']))

            # delta L(theta'(phi))
            weights_grad = torch.autograd.grad(
                val_loss, weights, retain_graph=True)[0]
            weights_grad_list.append(weights_grad)

        weights_grad = sum(weights_grad_list)

        if self._ren:
            return -weights_grad
        else:
            # update phi according grad of phi
            self._weights[ids] = weights.data / self._w_decay - weights_grad
            self._weights[ids] = torch.max(self._weights[ids], torch.ones_like(
                self._weights[ids]).fill_(EPSILON))

            return self._weights[ids].data

    def evaluate(self, set_type):
        self._model.eval()

        total_loss, total_non_padding = 0., 0.
        data_loader = self._data_loader[set_type]

        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        for batch in tqdm(data_loader,
                          desc="Evaluating {} set".format(set_type)):
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, segment_ids, label_ids = batch[:3]

            with torch.no_grad():
                logits, *_ = self._model(input_ids, token_type_ids=segment_ids)
                loss, non_padding = _compute_loss(logits, label_ids, criterion)
                total_loss += loss.item()
                total_non_padding += non_padding

        self._stat(total_loss / total_non_padding, 0, mode=set_type)
        return math.exp(min(total_loss / total_non_padding, 100))

    def save(self, epoch):
        torch.save(
            {k: (v.cpu() if v is not None else None)  # save to cpu tensors
             for k, v in self._model.state_dict().items()},
            os.path.join(self.save_dir, "pytorch_model_{}.bin".format(epoch)))
        if not os.path.exists(os.path.join(self.save_dir, CONFIG_NAME)):
            getattr(self._model, 'module', self._model).config.to_json_file(
                os.path.join(self.save_dir, CONFIG_NAME))
            self.tokenizer.save_vocabulary(self.save_dir)

        checkpoint = {
            "weights": self._weights,
            "args": self.args,
            "optim": self._optimizer.state_dict()
        }
        torch.save(
            checkpoint,
            os.path.join(self.save_dir, "checkpoint_{}.bin".format(epoch)))

    def _stat(self, loss, start, mode="training"):
        if "train" in mode:
            logger.info(
                ("Optimizer step %s; ppl: %5.2f; loss: %4.2f; lr: %7.5f; %6.0f sec")
                % (self.training_step,
                   math.exp(min(loss, 100)),
                   loss,
                   self._optimizer.param_groups[0]['lr'],
                   time.time() - start))
            self.writer.add_scalars("train_loss", {"loss_batch": loss}, self.training_step)
        else:
            logger.info(
                ("%s : PPL: %5.2f; loss: %4.2f") % (mode, math.exp(min(loss, 100)), loss))
            self.writer.add_scalars("{}_loss".format(mode), {"loss": loss}, self.training_step)
        sys.stdout.flush()

    def infer(self, data, outpath, has_gt=True):
        self._model.eval()

        predictions = []
        for session in tqdm(data, desc="Inferring test set"):
            with torch.no_grad():
                out_ids = self.sample_sequence(session, has_gt)
                out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                predictions.append(out_text)

        with open(outpath, 'w', encoding="UTF-8") as f:
            f.write("\n".join(predictions))

        gt = [self.tokenizer.decode(session[-1], skip_special_tokens=True)
              for session in data] if has_gt else None
        self._eval(predictions, gt)

    def sample_sequence(self, session, has_gt, current_output=None):
        if has_gt:
            session = session[:-1]
        if current_output is None:
            current_output = []

        for i in range(self.args.max_length):
            session.append(current_output)
            instance = build_one_json(session, self.tokenizer.eos_token_id, self._max_len)
            input_ids = torch.tensor(instance["input_ids_pad"],
                                     dtype=torch.long, device=self._device)
            token_type_ids = torch.tensor(instance["token_type_ids_pad"],
                                          dtype=torch.long, device=self._device)
            logits, *_ = self._model(input_ids, token_type_ids=token_type_ids)
            logits = logits[0, -1, :] / self.args.temperature
            logits = top_filtering(logits, top_k=self.args.top_k, top_p=self.args.top_p)
            probs = F.softmax(logits, dim=-1)

            prev = torch.multinomial(probs, 1)
            if i < self.args.min_length and prev.item() == self.tokenizer.eos_token_id:
                while prev.item() == self.tokenizer.eos_token_id:
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() == self.tokenizer.eos_token_id:
                break
            current_output.append(prev.item())

        return current_output

    def _eval(self, hypts, refs):
        hypts = [line.strip().split() for line in hypts]
        distinct = [round(x * 100, 4) for x in eval_distinct(hypts)]
        logger.info('distinct: {}'.format(distinct))
        if refs is not None:
            refs = [[line.strip().split()] for line in refs]
            gold_distinct = [round(x * 100, 4) for x in eval_distinct([x[0] for x in refs])]
            logger.info('gt distinct: {}'.format(gold_distinct))
            nltk_bleu = []
            chencherry = SmoothingFunction()
            for i in range(4):
                weights = [1 / (i + 1)] * (i + 1)
                nltk_bleu.append(
                    round(100 * corpus_bleu(
                        refs, hypts, weights=weights, smoothing_function=chencherry.method1), 4))
            logger.info('nltk BLEU: {}'.format(nltk_bleu))

    def infer_batch(self, data, outpath):
        self._model.eval()

        predictions = []
        for batch in tqdm(data_loader,
                          desc="Inferring {} set".format(set_type)):
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, segment_ids, label_ids = batch[:3]
            batch_size = label_ids.shape[0]
            with torch.no_grad():
                for i in range(batch_size):
                    inputs = (input_ids[i:i + 1], segment_ids[i:i + 1])
                    out_ids = self.sample_sequence(inputs)
                    out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                    predictions.append(out_text)

        with open(outpath, 'w', encoding="UTF-8") as f:
            f.write("\n".join(predictions))


def _get_optimizer(model, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln']
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optim.Adam(optimizer_grouped_parameters, lr=learning_rate)


def _set_lr(optimizer, step, schedule, lr,
            warmup_steps, warmup_proportion, n_embd, tot_steps,
            rate=0.5, decay_steps=10000, start_step=50000):
    if schedule == 'None':
        lr_this_step = lr
    elif schedule == 'noam':  # transformer like
        lr_this_step = lr * 1e4 * noam_decay(step + 1, warmup_steps, n_embd)
    elif schedule == 'noamwd':  # transformer like
        lr_this_step = lr * 1e4 * noamwd_decay(step + 1, warmup_steps, n_embd)
    elif schedule == 'exp':
        lr_this_step = lr * exponential_decay(step, rate, decay_steps, start_step)
    else:
        lr_this_step = lr * warmup_linear(step / tot_steps,
                                          warmup_proportion)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step


def _compute_loss(logits, label_ids, criterion, label_shape=False):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = label_ids[..., 1:].contiguous()
    # Flatten the tokens
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)),
                     shift_labels.view(-1))
    non_padding = shift_labels.ne(-1).sum().item()
    if label_shape:
        loss = loss.view(shift_labels.shape)
    return loss, non_padding


def _linear_normalize(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)
    return torch.zeros_like(weights)


def _softmax_normalize(weights):
    return nn.functional.softmax(weights, dim=0)


def _adam_delta(optimizer, model, grads):
    deltas = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            grad = grads[param]
            state = optimizer.state[param]

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            step = state['step'] + 1

            if group['weight_decay'] != 0:
                grad = grad + group['weight_decay'] * param.data

            exp_avg = exp_avg * beta1 + (1. - beta1) * grad
            exp_avg_sq = exp_avg_sq * beta2 + (1. - beta2) * grad * grad
            denom = exp_avg_sq.sqrt() + group['eps']

            bias_correction1 = 1. - beta1 ** step
            bias_correction2 = 1. - beta2 ** step
            step_size = group['lr'] * math.sqrt(
                bias_correction2) / bias_correction1

            deltas[param] = -step_size * exp_avg / denom

    param_to_name = {param: name for name, param in model.named_parameters()}

    return {param_to_name[param]: delta for param, delta in deltas.items()}


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec
