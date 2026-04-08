import torch
from torch.utils.backcompat import keepdim_warning

from config import *
from model import TransformerTranslationModel
from tokenizer import ZhTokenizer, EnTokenizer


def predict_batch(model, inputs, tokenizer):
    model.eval()

    with torch.no_grad():
        batch_size = inputs.shape[0]
        # 前向传播
        # 编码
        src_pad_mask = (inputs == model.zh_padding_idx).to(DEVICE)
        memory = model.encoder(inputs, src_pad_mask=src_pad_mask)
        # HERE 构建第一时间不的输入，长度为 N 的向量，内容全部为<sos> 的 id
        decoder_input = torch.full(size=(batch_size, 1), fill_value=tokenizer.sos_idx, device=DEVICE, dtype=torch.long)

        # 保存生成的结果
        generated_ids = []

        # 定义一个<eos> 的校验，循环中，判断是否已经生成<eos>
        is_finished = torch.full(size=[batch_size], fill_value=False).to(DEVICE)

        for i in range(MAX_SEQ_LEN):
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.size(1))
            # HERE 解码器前向 传播，得到解码输出 和隐藏状态 []
            decoder_outputs = model.decoder(decoder_input, memory,
                                            tgt_mask=tgt_mask,
                                            memory_key_padding_mask=src_pad_mask)  # [n,l,vocab_size]
            # 词选择策略：贪心解码
            next_token_ids = torch.argmax(decoder_outputs[:, -1, :], dim=-1, keepdim=True)  # [n,1,1]
            generated_ids.append(next_token_ids)  # [n,1,1]

            # 更新隐状态
            decoder_input = torch.cat([decoder_input, next_token_ids], dim=-1)
            # 是否生成<eos> 如果生成，退出循环
            is_finished = is_finished | (next_token_ids.squeeze(1) == tokenizer.eos_idx)

            if is_finished.all():
                break

    # 处理生成结果
    # 基于生成列表，将 id 转为 token
    generated_tensor = torch.cat(generated_ids, dim=1)  # [n,l]
    # 转为二维列表
    generate_list = generated_tensor.tolist()
    # 去除每行 eos 的内容
    for i, line in enumerate(generate_list):
        if tokenizer.eos_idx in line:
            line = line[:line.index(tokenizer.eos_idx)]
            generate_list[i] = line

    return generate_list


def predict():
    ch_tokenizer = ZhTokenizer.build_from_vocab(ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.build_from_vocab(EN_VOCAB_FILE)

    model = TransformerTranslationModel(ch_tokenizer.vocab_size, en_tokenizer.vocab_size, ch_tokenizer.pad_idx,
                                        en_tokenizer.pad_idx).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=DEVICE))

    ids = ch_tokenizer.encode("你好")
    inputs = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    result = predict_batch(model, inputs, en_tokenizer)
    print(en_tokenizer.decoder(result[0]))


if __name__ == '__main__':
    predict()
