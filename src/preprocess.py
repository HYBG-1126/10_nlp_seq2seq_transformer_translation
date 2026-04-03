from shapely.ops import orient
from sklearn.model_selection import train_test_split
from torch.distributed.elastic.multiprocessing.errors import record

from config import *
import pandas as pd

from tokenizer import EnTokenizer, ZhTokenizer


def preprocess():
    # HERE 读取数据
    df = pd.read_csv(RAW_FILE, sep='\t', usecols=[0, 1], names=['en', 'zh'], header=None, encoding="utf-8").dropna()

    #  HERE 划分数据
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)

    # HERE 构建词表
    EnTokenizer.build_vocab(df_train['en'].tolist(), EN_VOCAB_FILE)
    ZhTokenizer.build_vocab(df_train['zh'].tolist(), ZH_VOCAB_FILE)

    # HERE 获取 tokenizer
    en_tokenizer = EnTokenizer.build_from_vocab(EN_VOCAB_FILE)
    zh_tokenizer = ZhTokenizer.build_from_vocab(ZH_VOCAB_FILE)

    # HERE 构建数据集
    df_train['en'] = df_train['en'].apply(lambda x: en_tokenizer.encode(x, mark=True))
    df_train['zh'] = df_train['zh'].apply(lambda x: zh_tokenizer.encode(x))

    df_test['en'] = df_test['en'].apply(lambda x: en_tokenizer.encode(x, mark=True))
    df_test['zh'] = df_test['zh'].apply(lambda x: zh_tokenizer.encode(x))

    df_train.to_json(TRAIN_DATA_FILE, orient="records", lines=True)
    df_test.to_json(TEST_DATA_FILE, orient="records", lines=True)


if __name__ == '__main__':
    preprocess()
