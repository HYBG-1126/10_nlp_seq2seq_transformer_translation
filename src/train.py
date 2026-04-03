import time

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import  tqdm

from config import *
from dataset import get_dataloader
from model import TransformerTranslationModel
from tokenizer import ZhTokenizer, EnTokenizer


def train_one_epoch(model, optimizer, dataloader, loss_f):
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(DEVICE)  # HERE 输入数据
        targets = targets.to(DEVICE)

        decoder_input = targets[:, :-1]
        decoder_target = targets[:, 1:]

        # 元序列掩码
        src_pad_mask = (inputs != model.zh_embedding.padding_idx).to(DEVICE)  # [n,l]
        tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(DEVICE)

        decoder_output = model(src=inputs, tgt=decoder_target, src_pad_mask=src_pad_mask,
                               tgt_mask=tgt_mask)  # [n,l,vocab_size]

        loss_value = loss_f(decoder_output.transpose(1, 2), decoder_target)

        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss_value.item()
    return total_loss / len(dataloader)


def train():
    # 获取数据
    dataloader = get_dataloader(train=True)

    zh_tokenizer = ZhTokenizer.build_from_vocab(ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.build_from_vocab(EN_VOCAB_FILE)

    model = TransformerTranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
                                        en_vocab_size=en_tokenizer.vocab_size,
                                        zh_padding_idx=zh_tokenizer.pad_idx,
                                        en_padding_idx=en_tokenizer.pad_idx)

    model.to(DEVICE)
    loss_f = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_idx).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(LOG_DIR / time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    loss_min = float('inf')

    print("开始训练-----")

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch}")
        loss = train_one_epoch(model, optimizer, dataloader, loss_f)
        writer.add_scalar("Loss", loss, epoch)
        print(f"Epoch: {epoch}, Loss: {loss}")
        if loss_min > loss:
            torch.save(model.state_dict(), MODEL_DIR / f"model_{epoch}.pt")
            loss_min = loss

            print(f"保存模型：{epoch}")


if __name__ == '__main__':
    train()
