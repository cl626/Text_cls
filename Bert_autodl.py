import process_imdb
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import logging
import numpy as np
import pickle

# 使用BERT使其向量化

MAXLEN = 512 - 2
BATCHSIZE = 8

from transformers import BertForSequenceClassification, AdamW, BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup

# 调用词向量化模型
path = "bert-base-uncased"
with open('bert-base-uncased.pkl','rb') as f: 
    modelset = pickle.load(f)
tokenizer = modelset['tokenizer']

# tokenizer = BertTokenizer.from_pretrained(path)  #调用预训练模型
# path = "SamLowe/roberta-base-go_emotions"
# RLHFlow/ArmoRM-Llama3-8B-v0.1
# JiaqiLee/imdb-finetuned-bert-base-uncased

# 把文本转为序号
def convert_text_to_ids(tokenizer, sentence, limit_size=MAXLEN):
    t = tokenizer.tokenize(sentence)[:limit_size]       # 分词
    encoded_ids = tokenizer.encode(t)                   # 转为编码/词序号
    if len(encoded_ids) < limit_size + 2:               # 不够长补齐
        tmp = [0] * (limit_size + 2 - len(encoded_ids))
        encoded_ids.extend(tmp)
    return encoded_ids


'''构建数据集和迭代器'''
input_ids = [convert_text_to_ids(tokenizer, sen) for sen in process_imdb.train_samples]
# input_labels = process_imdb.get_onehot_labels(process_imdb.train_labels)
input_labels = torch.unsqueeze(torch.tensor(process_imdb.train_labels), dim=1)



# 空格映射为0，对应的掩码赋值为0，其余为1
def get_att_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [int(float(i > 0)) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks


atten_token_train = get_att_masks(input_ids)

'''构建数据集和数据迭代器，设定 batch_size 大小为'''

# print(atten_token_train[:5])
# import pdb; pdb.set_trace()

# 单词通过Berttokenizer转为词向量后，和标签一起传递给张量数据集TensorDataset，构建数据集，简化传统的Dataset的操作
train_set = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(atten_token_train),
                          torch.LongTensor(input_labels))
train_loader = DataLoader(dataset=train_set,
                          batch_size=BATCHSIZE,
                          shuffle=True
                          )

for i, (train, mask, label) in enumerate(train_loader):
    print(train.shape, mask.shape, label.shape)  ##torch.Size([8,512]) torch.Size([8,512]) torch.Size([8, 1])
    break

input_ids2 = [convert_text_to_ids(tokenizer, sen) for sen in process_imdb.test_samples]
input_labels2 = torch.unsqueeze(torch.tensor(process_imdb.test_labels), dim=1)
atten_tokens_eval = get_att_masks(input_ids2)
test_set = TensorDataset(torch.LongTensor(input_ids2), torch.LongTensor(atten_tokens_eval),
                         torch.LongTensor(input_labels2))
test_loader = DataLoader(dataset=test_set,
                         batch_size=BATCHSIZE, )

for i, (train, mask, label) in enumerate(test_loader):
    print(train.shape, mask.shape, label.shape)  #
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''预测函数，用于预测结果'''

def predict(logits):
    res = torch.argmax(logits, dim=1)  # 按行取每行最大的列下标
    return res


'''训练'''

def train_model(net, epoch=4):
    avg_loss = []
    net.train()  # 将模型设置为训练模式
    net.to(device)

    optimizer = AdamW(net.parameters(), lr=5e-5)

    accumulation_steps = 8
    for e in range(epoch):
        for batch_idx, (data, mask, target) in enumerate(train_loader):
            # optimizer.zero_grad()
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            output = net(data, token_type_ids=None, attention_mask=mask, labels=target)
            # logit是正负概率
            loss,logits=output[0],output[1]
            loss = loss / accumulation_steps  # 梯度积累
            avg_loss.append(loss.item())
            loss.backward()

            if ((batch_idx + 1) % accumulation_steps) == 0:
                # 每 8 次更新一下网络中的参数
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % 5 == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    e + 1, batch_idx, len(train_loader), 100. *
                    batch_idx / len(train_loader), np.array(avg_loss).mean()
                ))

    print('Finished Training')
    return net


def test_model(net):
    net.eval()
    net = net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, mask, label) in enumerate(test_loader):
            logging.info("test batch_id=" + str(batch_idx))

            data, mask, label = data.to(device), mask.to(device), label.to(device)
            output = net(data, token_type_ids=None, attention_mask=mask)  # 调用model模型时不传入label值。
            # output的形式为（元组类型，第0个元素是每个batch中好评和差评的概率）
            # print(output[0],label)
            print(predict(output[0]), label.flatten())
            total += label.size(0)  # 逐次按batch递增
            correct += (predict(output[0]) == label.flatten()).sum().item()
        print(f"正确分类的样本数 {correct}，总数 {total},准确率 {100.*correct/total:.3f}%")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

    with open('bert-base-uncased.pkl','rb') as f:
        modelset = pickle.load(f)
    pre_net = modelset['pre_net']
    # pre_net = BertForSequenceClassification.from_pretrained(path)
    params_dir = 'model/bert_base_model_beta.pkl'

    model = train_model(pre_net, epoch=4)
    torch.save(model.state_dict(), params_dir)  # 保存模型参数