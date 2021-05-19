import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TransformerModel
from dictionaries import special_tokens, PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN

PAD_INDEX = special_tokens.index(PAD_TOKEN)
SOS_INDEX = special_tokens.index(START_TOKEN)
EOS_INDEX = special_tokens.index(END_TOKEN)

''' This module will handle the text generation with beam search. '''


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model: TransformerModel, beam_size, max_seq_len,
            src_pad_idx=PAD_INDEX, trg_pad_idx=PAD_INDEX, trg_bos_idx=SOS_INDEX, trg_eos_idx=EOS_INDEX, alpha=1):

        super(Translator, self).__init__()

        self.alpha = alpha
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, targets, memory, memory_key_padding_mask):
        dec_output = self.model.decode(targets, memory, memory_key_padding_mask)
        return F.softmax(self.model.generate(dec_output), dim=-1)

    def _get_init_state(self, src_seq):
        beam_size = self.beam_size

        memory, memory_key_padding_mask = self.model.encode(src_seq)
        dec_output = self._model_decode(self.init_seq, memory, memory_key_padding_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        memory = memory.repeat(1, beam_size, 1)
        memory_key_padding_mask = memory_key_padding_mask.repeat(beam_size, 1)
        return memory, memory_key_padding_mask, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # done mask
        done = (gen_seq[:, step-1] == PAD_INDEX).logical_or(gen_seq[:, step-1] == EOS_INDEX).view(beam_size, 1)

        # Get k candidates for each beam, k^2 candidates in total.
        # dec_output: shape(Beam size, genSeqSize, vocabSize)
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # If token is not EOS or PADDING
        # Include the previous scores. Using broadCast
        # scores: shape(beam size, 1) best_k2_probs: shape(beam size, beam size)
        scores = torch.log(best_k2_probs).view(beam_size, -1).masked_fill(done, 0) + scores.view(beam_size, 1)
        # Notice: done sentence only count once for beam
        neg_inf = -10000
        # for done line where pos != 0 add inf to delete
        neg_inf_mask = (torch.arange(0, beam_size, device=best_k2_probs.device).repeat(beam_size, 1) != 0).logical_and(done)
        scores = scores + torch.zeros(neg_inf_mask.shape, device=neg_inf_mask.device).masked_fill(neg_inf_mask, neg_inf)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]
        done_idx = done[best_k_r_idxs]
        best_k_idx = best_k_idx.masked_fill(done_idx.view(beam_size), PAD_INDEX)

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        # when done, add pad index
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            memory, memory_key_padding_mask, \
            gen_seq, scores = self._get_init_state(src_seq)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], memory, memory_key_padding_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()