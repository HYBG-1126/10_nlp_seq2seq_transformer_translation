import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from config import EN_VOCAB_FILE, ZH_VOCAB_FILE, MODEL_DIR, DEVICE
from dataset import get_dataloader
from model import TransformerTranslationModel
from predict import predict_batch
from tokenizer import EnTokenizer, ZhTokenizer


def evaluate(model, test_loader):
    # 用列表记录参考译文和预测译文
    references = []
    predicts = []

    en_tokenizer = EnTokenizer.build_from_vocab(EN_VOCAB_FILE)
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="测试"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            # 前向传播得到一批结果
            batch_result = predict_batch(model, inputs, en_tokenizer)
            # 合并这一批结果到总列表
            predicts.extend(batch_result)
            # 合并这一批的参考译文
            references.extend([[target[1:target.index(en_tokenizer.eos_idx)]] for target in targets.tolist()])

    bleu_score = corpus_bleu(references, predicts)
    return bleu_score


if __name__ == '__main__':
    ch_tokenizer = ZhTokenizer.build_from_vocab(ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.build_from_vocab(EN_VOCAB_FILE)

    model = TransformerTranslationModel(ch_tokenizer.vocab_size, en_tokenizer.vocab_size, ch_tokenizer.pad_idx,
    en_tokenizer.pad_idx)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt"))
    test_data_loader = get_dataloader(train=False)
    print(evaluate(model, test_data_loader))
