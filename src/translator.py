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
        self.n_beam = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.tgt_sos_idx = trg_bos_idx
        self.tgt_eos_idx = trg_eos_idx
        self.tgt_pad_idx = trg_pad_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.tgt_sos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def get_model_output(self, sources, targets):
        memory, memory_key_padding_mask = self.model.encode(sources)
        dec_output = self.model.decode(targets, memory, memory_key_padding_mask)
        output = self.model.generate(dec_output)
        return memory.transpose(0, 1), dec_output, output

    def _model_decode(self, targets, memory, memory_key_padding_mask):
        dec_output = self.model.decode(targets, memory, memory_key_padding_mask)
        return F.softmax(self.model.generate(dec_output), dim=-1)

    def _get_init_state(self, src_seq):
        beam_size = self.n_beam

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

        beam_size = self.n_beam

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

    # def translate_sentence(self, src_seq):
    #     # Only accept batch size equals to 1 in this function.
    #     # TODO: expand to batch operation.
    #     assert src_seq.size(0) == 1
    #
    #     src_pad_idx, trg_eos_idx = self.src_pad_idx, self.tgt_eos_idx
    #     max_seq_len, beam_size, alpha = self.max_seq_len, self.n_beam, self.alpha
    #
    #     with torch.no_grad():
    #         memory, memory_key_padding_mask, \
    #         gen_seq, scores = self._get_init_state(src_seq)
    #
    #         ans_idx = 0  # default
    #         for step in range(2, max_seq_len):  # decode up to max length
    #             dec_output = self._model_decode(gen_seq[:, :step], memory, memory_key_padding_mask)
    #             gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)
    #
    #             # Check if all path finished
    #             # -- locate the eos in the generated sequences
    #             eos_locs = gen_seq == trg_eos_idx
    #             # -- replace the eos with its position for the length penalty use
    #             seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
    #             # -- check if all beams contain eos
    #             if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
    #                 # TODO: Try different terminate conditions.
    #                 _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
    #                 ans_idx = ans_idx.item()
    #                 break
    #     return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()

    def translate_sentence(self, source):
        batch_size = source.shape[0]
        device = source.device
        source_beam = source.repeat_interleave(self.n_beam, dim=0)

        with torch.no_grad():
            gen_seq = torch.full((batch_size, 1), self.tgt_sos_idx, dtype=torch.long, device=device)
            output_logits = self.model(source, gen_seq)

            best_k2_probs, best_k2_idx = output_logits[:, -1, :].topk(self.n_beam)

            # print("best_k2_probs", best_k2_probs, file=log_file)
            # print("best_k2_idx", best_k2_idx, file=log_file)

            scores = F.log_softmax(best_k2_probs, dim=-1).reshape(-1)

            # print("scores", scores, file=log_file)

            best_k2_idx = best_k2_idx.reshape(-1)

            done = best_k2_idx == self.tgt_eos_idx
            gen_seq = torch.cat((gen_seq.repeat_interleave(self.n_beam, dim=0), best_k2_idx.unsqueeze(dim=-1)), dim=-1)

            # print("gen_seq", gen_seq, file=log_file)

            for i in range(self.max_seq_len - 1):
                if done.sum().item() == batch_size * self.n_beam:
                    break

                output_logits = self.model(source_beam, gen_seq)
                best_k2_probs, best_k2_idx = output_logits[:, -1, :].topk(self.n_beam)

                # print("best_k2_probs", best_k2_probs, file=log_file)
                # print("best_k2_idx", best_k2_idx, file=log_file)

                scores = F.log_softmax(best_k2_probs, dim=-1).masked_fill(done[:, None], 0) + scores[:, None]

                # print("scores", scores, file=log_file)

                neg_inf_mask = torch.BoolTensor([False] + [True] * (self.n_beam - 1)).to(device).repeat(
                    batch_size * self.n_beam, 1).logical_and(done[:, None])
                scores = scores + torch.zeros(neg_inf_mask.shape, device=neg_inf_mask.device).masked_fill(neg_inf_mask,
                                                                                                          -1e6)

                # print("scores", scores, file=log_file)

                scores, best_k_idx_in_k2 = torch.topk(scores.view(batch_size, -1), self.n_beam, dim=-1, largest=True,
                                                      sorted=True)

                # print("scores", scores, file=log_file)
                # print("best_k_idx_in_k2", best_k_idx_in_k2, file=log_file)

                scores = scores.view(-1)
                best_k_idx = torch.gather(best_k2_idx.view(batch_size, -1), 1, best_k_idx_in_k2).view(-1)  # token
                best_r_idx = best_k_idx_in_k2 // self.n_beam \
                             + torch.arange(0, batch_size * self.n_beam, self.n_beam).to(device)[:, None]  # beam resort
                best_r_idx = best_r_idx.reshape(-1)

                done = done[best_r_idx]

                # print("best_k_idx", best_k_idx, file=log_file)
                # print("best_r_idx", best_r_idx, file=log_file)

                best_k_idx = best_k_idx.masked_fill(done, self.tgt_pad_idx)
                done = done.logical_or(best_k_idx == self.tgt_eos_idx)

                # print("best_k_idx", best_k_idx, file=log_file)
                # print("done", done, file=log_file)

                gen_seq = torch.cat((gen_seq[best_r_idx, :], best_k_idx.unsqueeze(dim=1)), dim=-1)
            # endfor

            eos_index = gen_seq == self.tgt_eos_idx
            has_eos, eos_index = eos_index.max(dim=-1)
            eos_index.masked_fill_(has_eos == 0, gen_seq.shape[-1])
            scores = scores / eos_index.pow(self.alpha)
            best_choice = scores.reshape(batch_size, self.n_beam).argmax(dim=-1) \
                          + torch.arange(0, batch_size * self.n_beam, self.n_beam).to(device)
            target = gen_seq[best_choice, :]
        return target.cpu().detach().numpy().tolist()
