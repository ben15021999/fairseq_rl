# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from fairseq import metrics, search, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch.distributions import Categorical

from fairseq.sequence_generator import SequenceGenerator
from fairseq.scoring import bleu
from omegaconf import II

@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion('reward_baseline_v2')
class RewardBaselineCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, max_order):
        super().__init__(task)
        self.tgt_dict = task.tgt_dict
        self.max_order = max_order
        self.scorer = bleu.Scorer(bleu.BleuConfig(pad=self.tgt_dict.pad(), eos=self.tgt_dict.eos(), unk=self.tgt_dict.unk()))
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # Targets for reward computation
        target = sample['target']

        # Forward encoder
        encoder_out = model.encoder(
            sample['net_input']['src_tokens'], src_lengths=sample['net_input']['src_lengths'],
            return_all_hiddens=True
        )

        device='cpu'

        # print(encoder_out)

        # Decode translations sequentially (greedy decoding)
        pred_toks, lprob_toks, lprobs, bas_toks = sequential_decoding(model, encoder_out, sample,
                                                                      max_len_decoding=100,
                                                                      device=device)
        # print(pred_toks)
        lprobs_added = lprob_toks.sum(axis=1)
        # lprobs_added = utils.move_to_cuda(lprobs_added)
        lprobs_avg = lprobs_added.mean()

        r_hat = torch.tensor([self.reword(target_i, y_g_i) for target_i, y_g_i in zip(target, pred_toks)])
        r_baseline = torch.tensor([self.reword(target_i, y_g_i) for target_i, y_g_i in zip(target, bas_toks)])

        rewards_detached = r_hat.detach()
        rewards_baseline_detached = r_baseline.detach()

        loss = (rewards_baseline_detached - rewards_detached) * lprobs_added
        loss = loss.sum()

        sample_size = sample['ntokens']
        logging_output = {
            'loss': loss.data,
            # 'ntokens': sample['ntokens'],
            'n_sentences': sample['target'].size(0),
            'sample_size': sample_size
        }
        # print(pred_toks)
        return loss, sample_size, logging_output

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--max_order', type=int, default=4, help='Order for bleu score (default: 4)')

    def reword(self, ref, pred):
        self.scorer.reset(one_init=True)
        eos_token_id = torch.tensor(self.tgt_dict.eos())
        new_pred = []
        for i in pred:
            if(i != eos_token_id):
                new_pred.append(i)
            else:
                new_pred.append(i)
                break
        pred = torch.tensor(new_pred)
        self.scorer.add(ref.type(torch.IntTensor), pred.type(torch.IntTensor))
        return self.scorer.score(self.max_order)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        # ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        # sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        n_sentences = sum(log.get('n_sentences', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / n_sentences, n_sentences, round=3)
        # if sample_size != ntokens:
        #     metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
        #     metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        # else:
        #     metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

def _forward_one(model, encoded_source, tokens, incremental_states=None, temperature=1., return_attn=False,
                 return_logits=False, **decoder_kwargs):
    if incremental_states is not None:
        decoder_out = list(model.decoder(tokens, encoded_source, incremental_state=incremental_states,
                                         **decoder_kwargs))
    else:
        decoder_out = list(model.decoder(tokens, encoded_source, **decoder_kwargs))
    decoder_out[0] = decoder_out[0][:, -1:, :].clone()
    # print(decoder_out[0].size())
    if temperature != 1.:
        decoder_out[0].div_(temperature)
    attn = decoder_out[1]
    if type(attn) is dict:
        attn = attn['attn'][0]
    attn = None
    if attn is not None:
        if type(attn) is dict:
            attn = attn['attn']
        attn = attn[:, :, -1, :]  # B x L x t
    if return_logits:
        logits_t = decoder_out[0][:, -1, :].clone()
        return logits_t, attn
    log_probs = model.get_normalized_probs(decoder_out, log_probs=True)
    log_probs = log_probs[:, -1, :].clone()
    return log_probs, attn


def sequential_decoding(model, encoded_source, sample, max_len_decoding, device):
    # model.eval()
    pred_toks_pred = []
    pred_toks_bas = []
    batch_size = len(encoded_source['encoder_out'][0])
    eos_token_id = torch.tensor(model.decoder.dictionary.eos()).to(device)
    pad_token_id = torch.tensor(model.decoder.dictionary.pad()).to(device)
    context_pred = sample['net_input']['prev_output_tokens'][:,:1]
    context_bas = sample['net_input']['prev_output_tokens'][:,:1]
    # print(context)
    states = {}
    lprob_toks_pred = []
    all_lprobs_pred = []
    masking_matrix_pred = []
    aux_masking_matrix_pred = []
    lprob_toks_bas = []
    all_lprobs_bas = []
    masking_matrix_bas = []
    aux_masking_matrix_bas = []

    for _ in range(max_len_decoding):
        # We need 2 sampling techniques
        lprobs_pred, attn_t_pred = _forward_one(model, encoded_source, context_pred, incremental_states=states)
        lprobs_bas, attn_t_bas = _forward_one(model, encoded_source, context_bas, incremental_states=states)
        # lprobs[:, pad_token_id] = -math.inf  # never select pad  (MAYBE I CAN ADD MIN LENGTH?)
        # print(lprobs.size())
        # Argmax
        pred_tok_bas = lprobs_bas.argmax(dim=1, keepdim=True)
        lprob_tok_bas = torch.gather(lprobs_bas, dim=1, index=pred_tok_bas)
        # Sampling
        dist = Categorical(logits=lprobs_pred)
        pred_tok_pred = dist.sample().unsqueeze(dim=1)
        # print(pred_tok_pred)
        lprob_tok_pred = torch.gather(lprobs_pred, dim=1, index=pred_tok_pred)
        # print(lprob_tok.size())
        # print(lprob_tok_index.size())
        # Check if predicted token is <eos>
        pred_token_bool = torch.where(pred_tok_pred == eos_token_id, torch.tensor(1.0).to(device),
                                      torch.tensor(0.0).to(device))
        bas_token_bool = torch.where(pred_tok_bas == eos_token_id, torch.tensor(1.0).to(device),
                                      torch.tensor(0.0).to(device))
        if len(aux_masking_matrix_pred) > 0:
            pred_token_bool = torch.logical_or(aux_masking_matrix_pred[-1], pred_token_bool)
            pred_token_bool = torch.where(pred_token_bool == True, torch.tensor(1.0).to(device),
                                          torch.tensor(0.0).to(device))
            see_if_previous_was_eos = torch.logical_or(masking_matrix_pred[-1], aux_masking_matrix_pred[-1]).to(device)
            pred_token_bool_true = torch.logical_and(see_if_previous_was_eos, pred_token_bool).to(device)
            masking_matrix_pred.append(pred_token_bool_true)
            # BASELINE
            bas_token_bool = torch.logical_or(aux_masking_matrix_bas[-1], bas_token_bool)
            bas_token_bool = torch.where(bas_token_bool == True, torch.tensor(1.0).to(device),
                                          torch.tensor(0.0).to(device))
            see_if_previous_was_eos_bas = torch.logical_or(masking_matrix_bas[-1],
                                                           aux_masking_matrix_bas[-1]).to(device)
            bas_token_bool_true = torch.logical_and(see_if_previous_was_eos_bas, bas_token_bool).to(device)
            masking_matrix_bas.append(bas_token_bool_true)
        else:
            masking_matrix_pred.append(torch.zeros(pred_token_bool.size()).to(device))
            masking_matrix_bas.append(torch.zeros(bas_token_bool.size()).to(device))
        aux_masking_matrix_pred.append(pred_token_bool)
        aux_masking_matrix_bas.append(bas_token_bool)

        pred_toks_pred.append(pred_tok_pred)
        pred_toks_bas.append(pred_tok_bas)
        context_pred = torch.cat((context_pred, pred_tok_pred), 1)
        context_bas = torch.cat((context_bas, pred_tok_bas), 1)
        lprob_toks_pred.append(lprob_tok_pred)
        all_lprobs_pred.append(lprobs_pred)
        lprob_toks_bas.append(lprob_tok_bas)
        all_lprobs_bas.append(lprobs_bas)
        count_token_pred = pred_token_bool[pred_token_bool == 0].size()[0]
        count_token_bas = bas_token_bool[bas_token_bool == 0].size()[0]
        count_token = count_token_pred + count_token_bas
        if count_token == 0:
            break

    # for tok in pred_toks:
    #     print(model.decoder.dictionary.__getitem__(tok[0]))
    masking_matrix_pred = torch.cat(masking_matrix_pred, 1)
    pred_toks_pred = torch.cat(pred_toks_pred, 1)
    lprob_toks_pred = torch.cat(lprob_toks_pred, 1)
    all_lprobs_pred = torch.stack(all_lprobs_pred, 1)
    # BASELINE
    masking_matrix_bas = torch.cat(masking_matrix_bas, 1)
    pred_toks_bas = torch.cat(pred_toks_bas, 1)

    # Apply masking (padding tokens after the <eos> token.)
    pred_toks_pred[masking_matrix_pred == 1.0] = pad_token_id
    pred_toks_bas[masking_matrix_bas == 1.0] = pad_token_id
    # print(pred_toks[0,:])
    # Apply masking (set probability values to zero)
    all_lprobs_pred[masking_matrix_pred == 1.0] = torch.zeros(all_lprobs_pred.size()[-1]).to(device)

    return pred_toks_pred, lprob_toks_pred, all_lprobs_pred, pred_toks_bas