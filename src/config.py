from pathlib import Path

import torch

# ROOT_DIR
ROOT_DIR = Path(__file__).parent.parent

# DATA_DIR
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
# FILES
RAW_FILE = RAW_DIR / "cmn.txt"
TRAIN_DATA_FILE = PROCESSED_DIR / "train.jsonl"
TEST_DATA_FILE = PROCESSED_DIR / "test.jsonl"
EN_VOCAB_FILE = PROCESSED_DIR / "en_vocab.txt"
ZH_VOCAB_FILE = PROCESSED_DIR / "zh_vocab.txt"

# MODEL_DIR
MODEL_DIR = ROOT_DIR / "models"

# LOG_DIR
LOG_DIR = ROOT_DIR / "logs"

# 特殊 token
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

# 超参数
MAX_SEQ_LEN = 128  # 最长序列长度
BATCH_SIZE = 64  # 批次大小
EPOCHS = 50  # 训练轮数
LEARNING_RATE = 1e-3  # 学习率
SEED = 42  # 随机数种子

# 模型结构参数
DIM_MODEL = 128
NUM_HEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

# 设备
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print(ROOT_DIR)