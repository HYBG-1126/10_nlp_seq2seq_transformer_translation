from config import *
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer


class BaseTokenizer:
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2idx = {word: idx for idx, word in enumerate(vocab_list)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab_list)}
        self.unk_idx = self.word2idx[UNK_TOKEN]
        self.pad_idx = self.word2idx[PAD_TOKEN]
        self.sos_idx = self.word2idx[SOS_TOKEN]
        self.eos_idx = self.word2idx[EOS_TOKEN]

    @classmethod
    def tokenize(cls, text) -> list[str]:
        """
        将文本转换为索引序列
        :param text: 文本
        :return: 索引序列
        """
        pass

    def encode(self, text, mark=False):
        """
        将文本转换为索引序列
        :param text: 输入文本
        :return: 索引序列
        """
        tokens = self.tokenize(text)
        if mark:
            tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]

        return [self.word2idx.get(token, self.unk_idx) for token in tokens]

    # HERE 创建词表
    @classmethod
    def build_vocab(cls, sentences,file_path):
        vocab_set = set()
        for sentence in sentences:
            tokens = cls.tokenize(sentence)
            vocab_set.update(tokens)

        vocab_list = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(vocab_set)

        print("词表大小：", len(vocab_list))

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))

    @classmethod
    def build_from_vocab(cls, vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_list = f.read().split("\n")

        return cls(vocab_list)


class EnTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(self, text):
        return TreebankWordTokenizer().tokenize(text)

    def decoder(self, tokens):
        return TreebankWordDetokenizer().detokenize([self.idx2word[idx] for idx in tokens])


class ZhTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return list(text)


if __name__ == '__main__':
    pass
