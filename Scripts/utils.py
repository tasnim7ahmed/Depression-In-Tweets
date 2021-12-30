import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score

from common import get_parser
from model import BertFGBC, RobertaFGBC, XLNetFGBC, DistilBertFGBC
from dataset import DatasetBert, DatasetRoberta, DatasetXLNet, DatasetDistilBert

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_device():
    device = ""
    if(args.device=="cpu"):
        device = "cpu"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(device=="cpu"):
            print("GPU not available.")
    return device

def sorting_function(val):
    return val[1]    

def load_prediction():
    bert_path = (f'{args.output_path}bert-base-uncased---test_acc---0.9260377358490566.csv')
    xlnet_path = (f'{args.output_path}xlnet-base-cased---test_acc---0.949685534591195.csv')
    roberta_path = (f'{args.output_path}roberta-base---test_acc---0.949685534591195.csv')
    distilbert_path = (f'{args.output_path}distilbert-base-uncased---test_acc---0.9275471698113208.csv')

    bert = pd.read_csv(bert_path)
    xlnet = pd.read_csv(xlnet_path)
    roberta = pd.read_csv(roberta_path)
    distilbert = pd.read_csv(distilbert_path)

    return bert, xlnet, roberta, distilbert

def print_stats(max_vote_df, bert, xlnet, roberta, distilbert):
    print(max_vote_df.head())
    print(f'---Ground Truth---\n{bert.target.value_counts()}')
    print(f'---Bert---\n{bert.y_pred.value_counts()}')
    print(f'---XLNet---\n{xlnet.y_pred.value_counts()}')
    print(f'---Roberta---\n{roberta.y_pred.value_counts()}')
    print(f'---DistilBert---\n{distilbert.y_pred.value_counts()}')

def evaluate_ensemble(max_vote_df):
    y_test = max_vote_df['target'].values
    y_pred = max_vote_df['pred'].values
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print('Accuracy:', acc)
    print('Mcc Score:', mcc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_score:', f1)
    print('classification_report: ', classification_report(y_test, y_pred, digits=4))
    
    max_vote_df.to_csv(f'{args.output_path}Ensemble-{args.ensemble_type}---test_acc---{acc}.csv', index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)

def generate_dataset_for_ensembling(pretrained_model, df):
    if(pretrained_model == "bert-base-uncased"):
        dataset = DatasetBert(text=df.text.values, target=df.target.values, pretrained_model="bert-base-uncased")
    elif(pretrained_model== "roberta-base"):
        dataset = DatasetRoberta(text=df.text.values, target=df.target.values, pretrained_model="roberta-base")
    elif(pretrained_model== "xlnet-base-cased"):
        dataset = DatasetXLNet(text=df.text.values, target=df.target.values, pretrained_model="xlnet-base-cased")
    elif(pretrained_model == "distilbert-base-uncased"):
        dataset = DatasetDistilBert(text=df.text.values, target=df.target.values, pretrained_model="distilbert-base-uncased")

    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = args.test_batch_size,
        shuffle = False
    )

    return data_loader

def load_models():
    bert_path = (f'{args.model_path}bert-base-uncased_Best_Val_Acc.bin')
    xlnet_path = (f'{args.model_path}xlnet-base-cased_Best_Val_Acc.bin')
    roberta_path = (f'{args.model_path}roberta-base_Best_Val_Acc.bin')
    distilbert_path = (f'{args.model_path}distilbert-base-uncased_Best_Val_Acc.bin')

    bert = BertFGBC(pretrained_model="bert-base-uncased")
    xlnet = XLNetFGBC(pretrained_model="xlnet-base-cased")
    roberta = RobertaFGBC(pretrained_model="roberta-base")
    distilbert = DistilBertFGBC(pretrained_model="distilbert-base-uncased")

    bert.load_state_dict(torch.load(bert_path))
    xlnet.load_state_dict(torch.load(xlnet_path))
    roberta.load_state_dict(torch.load(roberta_path))
    distilbert.load_state_dict(torch.load(distilbert_path))

    return bert, xlnet, roberta, distilbert