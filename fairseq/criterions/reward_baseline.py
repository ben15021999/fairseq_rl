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

from fairseq.sequence_generator import SequenceGenerator
from fairseq.scoring import bleu
from omegaconf import II

@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion('reward_baseline')
class RewardBaselineCriterion(FairseqCriterion):

    def __init__(self, task, sampling_topk, sentence_avg, beam_size, max_order):
        super().__init__(task)
        self.sampling_k = sampling_topk
        self.pad = task.tgt_dict.pad()
        self.beam_size = beam_size
        tgt_dict = task.tgt_dict
        self.max_order = max_order
        self.scorer = bleu.Scorer(bleu.BleuConfig(pad=tgt_dict.pad(), eos=tgt_dict.eos(), unk=tgt_dict.unk()))
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # model.eval()
        # src_dict = self.task.source_dictionary
        # tgt_dict = self.task.target_dictionary
        # self.sample_gen = SequenceGenerator([model], tgt_dict, beam_size=self.n_sample)
        # self.greedy_gen = SequenceGenerator([model], tgt_dict, beam_size=1)
        # net_output = model(**sample['net_input'])
        # loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        # sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        # logging_output = {
        #     'loss': loss.data,
        #     'ntokens': sample['ntokens'],
        #     'nsentences': sample['target'].size(0),
        #     'sample_size': sample_size,
        # }
        # return loss, sample_size, logging_output
        tgt_dict = self.task.target_dictionary
        search_strategy = search.Sampling(tgt_dict, sampling_topk=self.sampling_k)
        self.sample_gen = SequenceGenerator([model], tgt_dict, beam_size=self.beam_size, search_strategy=search_strategy)
        self.greedy_gen = SequenceGenerator([model], tgt_dict, beam_size=1)
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--sampling_topk', type=int, default=2, help='Number of sample size (default: 5)')
        parser.add_argument('--beam_size', type=int, default=4, help='Beam size (default: 4)')
        parser.add_argument('--max_order', type=int, default=4, help='Order for bleu score (default: 4)')

    def reword(self, ref, pred):
        self.scorer.reset(one_init=True)
        self.scorer.add(ref.type(torch.IntTensor), pred.type(torch.IntTensor))
        return self.scorer.score(self.max_order)

    def compute_loss(self, model, net_output, sample, reduce=True):
        # Generate baseline/samples
        s = utils.move_to_cuda(sample)
        with torch.no_grad():
            y_g = self.greedy_gen.generate([model], sample)
            y_hat = self.sample_gen.generate([model], sample)
        # print(y_hat)
        ref = s['target']
        model.train()
        # rewords
        r_g = torch.tensor([self.reword(ref_i, y_g_i[0]['tokens']) for ref_i, y_g_i in zip(ref, y_g)])
        r_hat = torch.tensor([[self.reword(ref_i, y_hat_i_n['tokens']) for y_hat_i_n in y_hat_i] for ref_i, y_hat_i in zip(ref, y_hat)])
        r_d = r_hat - r_g.unsqueeze(-1)
        # print(r_d)

        # scores
        net_input = {
            'src_tokens': s['net_input']['src_tokens'],
            'src_lengths': s['net_input']['src_lengths'],
        }
        encoder_out = model.encoder(**net_input)
        bos = s['net_input']['prev_output_tokens'][:,:1]

        scores = []
        for n in range(self.beam_size):
            output_tokens = [y_hat_i[n]['tokens'] for y_hat_i in y_hat]
            output_tokens = rnn_utils.pad_sequence(output_tokens, batch_first=True, padding_value=self.pad)

            prev_output_tokens = torch.cat([bos, output_tokens], dim=-1)
            net_output = model.decoder(prev_output_tokens, encoder_out=encoder_out)
            
            lprobs = model.get_normalized_probs(net_output, log_probs=True)[:, :-1, :]
            lprobs = lprobs.reshape(-1, lprobs.size(-1))
            lprobs = -lprobs[range(lprobs.size(0)), output_tokens.reshape(-1)]
            lprobs = lprobs.reshape(output_tokens.size())
            # lprobs = -lprobs.gather(dim=-1)
            lprobs = lprobs.sum(dim=-1, keepdim=True)
            # print(lprobs)
            scores.append(lprobs)
        
        scores = torch.cat(scores, dim=-1)
        r_d = r_d.to(scores.device)

        loss = ((scores * r_d) / self.beam_size).sum()

        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True