import math

import torch
import torch.nn as nn
from nltk.misc.wordfinder import step

from config import *
from tokenizer import ZhTokenizer, EnTokenizer


class PositionalEncoding(nn.Module):
    # 生成位置编码矩阵
    def __init__(self):
        super().__init__()
        self.dim_model = DIM_MODEL
        self.max_seq_len = MAX_SEQ_LEN
        # 初始矩阵
        self.pe = torch.zeros(size=(self.max_seq_len, self.dim_model), dtype=torch.float).to(DEVICE)

        # 便利每一行
        for row in range(self.max_seq_len):
            # 遍历每一列
            for col in range(0, self.dim_model, 2):
                self.pe[row, col] = math.sin(row / (10000 ** (col / self.dim_model)))
                self.pe[row, col + 1] = math.cos(row / (10000 ** (col / self.dim_model)))

        self.register_buffer("myPe", self.pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class TransformerTranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_idx, en_padding_idx):
        super().__init__()
        # 词嵌入
        self.zh_embedding = nn.Embedding(zh_vocab_size, DIM_MODEL, padding_idx=zh_padding_idx)
        self.en_embedding = nn.Embedding(en_vocab_size, DIM_MODEL, padding_idx=en_padding_idx)
        # 位置编码
        self.position_encoding = PositionalEncoding()

        self.transformer = nn.Transformer(
            d_model=DIM_MODEL,
            nhead=NUM_HEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_ENCODER_LAYERS,
            batch_first=True,
        )
        self.linear = nn.Linear(DIM_MODEL, en_vocab_size)

    def forward(self, src, tgt, src_pad_mask, tgt_mask):
        # 输入序列
        # 编码
        memory = self.encoder(src, src_pad_mask)
        # 解码
        output = self.decoder(tgt, memory, tgt_mask, src_pad_mask)
        return output

    def encoder(self, src, src_pad_mask):
        # HERE 输入原序列
        # 词嵌入
        embed = self.zh_embedding(src)  # [batch_size, seq_len, dim_model]
        # 叠加位置编码
        input = self.position_encoding(embed)  # [batch_size, seq_len, dim_model]
        # 编码器前向传播
        memory = self.transformer.encoder(src=input, src_key_padding_mask=src_pad_mask)

        return memory

    def decoder(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        # HERE 输入目标序列
        # 词嵌入
        embed = self.en_embedding(tgt)  # [batch_size, seq_len, dim_model]
        # 叠加位置编码
        input = self.position_encoding(embed)  # [batch_size, seq_len, dim_model]
        # 解码器前向传播
        output = self.transformer.decoder(tgt=input, memory=memory, tgt_mask=tgt_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)  # [batch_size, seq_len, dim_model]
        # 线性映射
        output = self.linear(output)  # [batch_size, seq_len, en_vocab_size]
        return output


if __name__ == '__main__':
    pass
