import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, RobertaModel, XLNetModel, DistilBertModel

from common import get_parser

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class BertFGBC(nn.Module):
    def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.Bert = BertModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.bert_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _,last_hidden_state = self.Bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        #print(f'Last Hidden State - {last_hidden_state.shape}')
        bo = self.drop1(last_hidden_state)
        #print(f'Dropout1 - {bo.shape}')
        bo = self.linear(bo)
        #print(f'Linear1 - {bo.shape}')
        bo = self.batch_norm(bo)
        #print(f'BatchNorm - {bo.shape}')
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)
        #print(f'Dropout2 - {bo.shape}')

        output = self.out(bo)
        #print(f'Output - {output.shape}')
        return output

class RobertaFGBC(nn.Module):
    def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.Roberta = RobertaModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.roberta_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        _,last_hidden_state = self.Roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        bo = self.drop1(last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output

class DistilBertFGBC(nn.Module):
    def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.DistilBert = DistilBertModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.distilbert_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.DistilBert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        
        bo = self.drop1(mean_last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output

    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

class XLNetFGBC(nn.Module):
    def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.XLNet = XLNetModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.xlnet_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state = self.XLNet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)

        bo = self.drop1(mean_last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output
        
    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state