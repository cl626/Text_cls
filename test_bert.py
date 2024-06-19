import torch
import Bert_autodl
import os
from transformers import BertForSequenceClassification, AdamW,BertTokenizer,BertModel
 # 测试
if __name__=="__main__":
    params_dir='model/bert_base_model_test.pkl'

    path='/data/yanghan/Bert_related/bert_base_uncased/'
    model=BertForSequenceClassification.from_pretrained(path)
    model.load_state_dict(torch.load(params_dir))
    Bert_autodl.test_model(model)
