from distutils.util import strtobool
from email.policy import default
import math
from operator import mod
from xmlrpc.client import boolean
import torch.nn.functional as F
import collections
import torch
import numpy
import logging
from collections import defaultdict

from fairseq import utils, metrics, search, scoring
from fairseq.sequence_generator import SequenceGenerator
from fairseq.scoring import bleu

from fairseq.criterions import FairseqCriterion, register_criterion

logger = logging.getLogger(__name__)

@register_criterion('reward_shaping')
class RewardShaping(FairseqCriterion):
    def __init__(self, task, beam_size, multinomial_sample_train, sampling_topk, max_order):
        super().__init__(task)
        self.beam_size = beam_size
        self.multinomial_sample_train = multinomial_sample_train
        self.max_order = max_order
        tgt_dict = task.tgt_dict
        self.scorer = bleu.Scorer(bleu.BleuConfig(pad=tgt_dict.pad(), eos=tgt_dict.eos(), unk=tgt_dict.unk()))
        self.sampling_topk = sampling_topk
        # print(modgleu)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--beam_size', default='5', type=int, metavar='D',
                            help='Beam size')
        parser.add_argument('--multinomial_sample_train', default='True', type=bool, metavar='D',
                            help="Multinomial Sample Train")
        parser.add_argument('--sampling_topk', default='2', type=int, metavar='D',
                            help="Top-K sampling")
        parser.add_argument('--max_order', default='4', type=int, metavar='D',
                            help='Max order')

    def forward(self, model, sample, reduce=True):
        # sample mode
        #print('!!!RL loss.')
        
        # src_dict = self.task.source_dictionary
        tgt_dict = self.task.target_dictionary
        eos_idx = self.task.target_dictionary.eos()
        sample_beam = self.beam_size
        search_strategy = (
            search.Sampling(tgt_dict, sampling_topk=self.sampling_topk) if self.multinomial_sample_train else None
        )
        translator = SequenceGenerator([model], tgt_dict=tgt_dict,
                                       beam_size=sample_beam, search_strategy=search_strategy)
        translator.cpu()
        ct = 0
        translations = []

        s = utils.move_to_cuda(sample)
        input = s['net_input']
        bos = s['net_input']['prev_output_tokens'][:,:1]
        with torch.no_grad():
            hypos = translator.generate(
                [model],
                sample,
            )
        for i, id in enumerate(s['id'].data):
            src = input['src_tokens'].data[i, :]
            # remove padding from ref
            ref = utils.strip_pad(s['target'].data[i, :], tgt_dict.pad()) if s['target'] is not None else None
            translations.append((id, src, ref, hypos[i]))
            ct += 1
        # print("sample batch size:", ct)
        model.train()

        # MLE loss
        mle_net_output = model(**sample['net_input'])
        mle_lprobs = model.get_normalized_probs(mle_net_output, log_probs=True)
        mle_lprobs = mle_lprobs.view(-1, mle_lprobs.size(-1))
        mle_target = model.get_targets(sample, mle_net_output).view(-1)
        mle_loss = F.nll_loss(
            mle_lprobs, 
            mle_target,
            ignore_index=self.padding_idx, 
            reduction='sum' if reduce else None)
        mle_tokens = sample['ntokens']
        avg_mle_loss = mle_loss / mle_tokens
        print('avg_mle_loss:', avg_mle_loss)

        # RL loss
        batch_rl_loss = 0
        batch_tokens = 0
        sample_ind = 0
        for sample_id, src_tokens, tgt_tokens, hypos in translations:
            # calculate bleu
            sample_ind += 1
            # rewards = torch.Tensor(sample_beam).float().cuda()
            # logprobs = torch.Tensor(sample_beam).float().cuda()
            reward = torch.tensor(100.0)
            for i in range(sample_beam):
                hypo = hypos[i]
                trans_tokens = hypo['tokens']
                self.scorer.add(tgt_tokens.type(torch.IntTensor), trans_tokens.type(torch.IntTensor))
                terminal_reward = self.scorer.score(self.max_order)
                if torch.gt(terminal_reward, reward):
                    reward = terminal_reward
                # rewards[i] = self.compute_gleu(tgt_tokens.cpu(), trans_tokens.cpu(), max_order=self.max_order, gram=self.gram).cuda()
                # one_sample loss calculation
            tgt_input_tokens = trans_tokens.new(trans_tokens.shape).fill_(0)
            assert trans_tokens[-1] == eos_idx
            tgt_input_tokens[0] = eos_idx
            tgt_input_tokens[1:] = trans_tokens[:-1]
            train_sample = {
                'net_input': {
                    'src_tokens': src_tokens.view(1, -1),
                    'src_lengths': torch.LongTensor(src_tokens.numel()).view(1, -1),
                    'prev_output_tokens': tgt_input_tokens.view(1, -1),
                },
                'target': trans_tokens.view(1, -1)
            }
            train_sample = utils.move_to_cpu(train_sample)
            net_output = model(**train_sample['net_input'])
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(train_sample, net_output).view(-1, 1)
            non_pad_mask = target.ne(tgt_dict.pad())
            lprob = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
            # ntokens = len(train_sample['target'])
            # batch_tokens += ntokens
            # rewards = rewards[::-1].cumsum()[::-1]
            rl_loss = torch.sum(lprob * reward)  # one sample loss   
            batch_rl_loss += rl_loss
        # print('avg_rl_loss:', avg_rl_loss)

        # if self.mle_weight:
        #     total_loss = self.mle_weight * avg_mle_loss + self.rl_weight * avg_rl_loss
        #     total_tokens = batch_tokens + mle_tokens
        # else:
        #     total_loss = avg_rl_loss
        #     total_tokens = batch_tokens

        logging_output = {
            'loss': utils.item(batch_rl_loss.data),
            'ntokens': batch_tokens,
            'nsentences': sample['target'].size(0),
            'sample_size': batch_tokens,
        }

        # print('total: ',total_loss)

        return batch_rl_loss, batch_tokens, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    # def _get_ngrams(self, segment, max_order):
    #     ngram_counts = collections.Counter()
    #     for order in range(1, max_order + 1):
    #         for i in range(0, len(segment) - order + 1):
    #             ngram = tuple(segment[i:i + order])
    #             ngram_counts[ngram] += 1
    #     return ngram_counts

    # def compute_gleu(self, reference_corpus, translation_corpus, max_order=4, gram=0, smooth=False):
    #     scores = torch.zeros(max_order)
    #     reference_array = numpy.array(reference_corpus)
    #     translation_array = numpy.array(translation_corpus)
    #     matches_by_order = [0] * max_order
    #     possible_matches_by_order_ref = [0] * max_order
    #     possible_matches_by_order_trans = [0] * max_order
    #     reference_length = 0
    #     translation_length = 0
    #     reference_length += reference_array.shape[0]
    #     translation_length += translation_array.shape[0]
    #     merged_ref_ngram_counts = collections.Counter()
    #     merged_ref_ngram_counts |= self._get_ngrams(reference_array, max_order)
    #     translation_ngram_counts = self._get_ngrams(translation_array, max_order)
    #     overlap = translation_ngram_counts & merged_ref_ngram_counts
    #     for ngram in overlap:
    #         matches_by_order[len(ngram) - 1] += overlap[ngram]
    #     for order in range(1, max_order + 1):
    #         possible_matches_trans = translation_length - order + 1
    #         if possible_matches_trans > 0:
    #             possible_matches_by_order_trans[order - 1] += possible_matches_trans
    #         possible_matches_ref = reference_length - order + 1
    #         if possible_matches_ref > 0:
    #             possible_matches_by_order_ref[order-1] += possible_matches_ref
    #     precisions = [0] * max_order
    #     recalls = [0] * max_order

    #     for i in range(0, max_order):
    #         if smooth:
    #             precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order_trans[i] + 1.))
    #             recalls[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order_ref[i] + 1.))
    #         else:
    #             if possible_matches_by_order_trans[i] > 0:
    #                 precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order_trans[i])
    #             else:
    #                 precisions[i] = 0.0
                
    #             if possible_matches_by_order_ref[i] > 0:
    #                 recalls[i] = (float(matches_by_order[i]) / possible_matches_by_order_ref[i])
    #             else:
    #                 recalls[i] = 0.0
    #     for i in range(max_order):
    #         scores[i] = min(precisions[i],recalls[i])

    #     if self.modgleu:
    #         if reference_length < max_order and translation_length < max_order:
    #             order = max(reference_length, translation_length)
    #             scores = scores[0:order]
    #         else:
    #             order = max_order
    #     else:
    #         order = max_order
    #     if gram == 0:
    #         if min(scores) > 0:
    #             log_scores = torch.log(scores)
    #             p_log_sum = torch.sum((1. / order) * log_scores)
    #             geo_mean = torch.exp(p_log_sum)
    #             return geo_mean.clone().detach()
    #         else:
    #             return torch.tensor(0.0)
    #     else:
    #         if scores[gram] > 0:
    #             return scores[gram].clone().detach()
    #         else:
    #             return torch.tensor(0.0)