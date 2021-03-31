import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform
from embedding import PositionalEmbedding
from utils.pad import pad_masking, subsequent_masking

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_len, tgt_vocab_len, d_model, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer, \
                TransformerDecoder, TransformerDecoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'transformer'
        self.d_model = d_model

        self.src_embedding = PositionalEmbedding(src_vocab_len, d_model)
        self.tgt_embedding = PositionalEmbedding(tgt_vocab_len, d_model)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        decoder_layers = TransformerDecoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        # 生成概率输出
        self.generator = nn.Linear(d_model, tgt_vocab_len)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform(p)

    def encode(self, sources):
        # embedding
        src = self.src_embedding(sources)  # (N, S, E)
        src = src.transpose(0, 1)  # (S, N, E)
        # get mask
        batch_size, sources_len = sources.size()
        src_key_padding_mask = pad_masking(sources, sources_len)  # (N, S)
        memory_key_padding_mask = src_key_padding_mask  # (N, S)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return memory, memory_key_padding_mask

    def decode(self, targets, memory, memory_key_padding_mask):
        # (N, T)
        batch_size, targets_len = targets.size()
        # embedding
        tgt = self.tgt_embedding(targets)
        tgt = tgt.transpose(0, 1)  # (T, N, E)
        # get mask
        tgt_key_padding_mask = pad_masking(targets, targets_len)  # (N, T)
        tgt_mask = subsequent_masking(targets)
        dec_output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask)  # (T, N, E)
        return dec_output

    def generate(self, dec_output):
        output = dec_output.transpose(0, 1)  # (N, T, E)
        output = self.generator(output)  # (N, T, FS)
        return output

    def forward(self, sources, targets) -> Tensor:
        r"""
        :param sources: (N, S)
        :param targets: (N, T)
        :return:    (N, T, FS)

        FS: targets词汇表的大小
        T: targets的长度
        S: src的长度
        """
        memory, memory_key_padding_mask = self.encode(sources)
        dec_output = self.decode(targets, memory, memory_key_padding_mask)
        output = self.generate(dec_output)
        
        # embedding
        # src = self.src_embedding(sources)  # (N, S, E)
        # tgt = self.tgt_embedding(targets)  # (N, T, E)
        # src = src.transpose(0, 1)  # (S, N, E)
        # tgt = tgt.transpose(0, 1)  # (T, N, E)
        #
        # if src.size(1) != tgt.size(1):
        #     raise RuntimeError("the batch number of src and tgt must be equal")
        #
        # if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
        #     raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        #
        # # get mask
        # batch_size, sources_len = sources.size()
        # batch_size, targets_len = targets.size()
        # # False不变 True Mask
        # src_key_padding_mask = pad_masking(sources, sources_len)  # (N, S)
        # tgt_key_padding_mask = pad_masking(targets, targets_len)  # (N, T)
        # memory_key_padding_mask = src_key_padding_mask  # (N, S)
        # tgt_mask = subsequent_masking(targets)  # (T, T)
        #
        # memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        # output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask,
        #                                   tgt_key_padding_mask=tgt_key_padding_mask,
        #                                   memory_key_padding_mask=memory_key_padding_mask)  # (T, N, E)
        # output = output.transpose(0, 1)  # (N, T, E)
        # output = self.generator(output)  # (N, T, FS)
        return output
