import torch.nn as nn
from config import *
from tokenizer import ZhTokenizer, EnTokenizer


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_model = DIM_MODEL
        self.max_seq_len = MAX_SEQ_LEN


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
        pass


if __name__ == '__main__':
    pass
