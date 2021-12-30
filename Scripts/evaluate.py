from operator import index
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score

from engine import test_eval_fn
from common import get_parser
from utils import set_device, load_models, generate_dataset_for_ensembling

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def test_evaluate(test_df, test_data_loader, model, device, pretrained_model = args.pretrained_model):
    print(f'\nEvaluating: ---{pretrained_model}---\n')
    y_pred, y_test = test_eval_fn(test_data_loader, model, device, pretrained_model)
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
    test_df['y_pred'] = y_pred
    pred_test = test_df[['text', 'label', 'target', 'y_pred']]
    pred_test.to_csv(f'{args.output_path}{pretrained_model}---test_acc---{acc}.csv', index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)

def evaluate_all_models():
    bert, xlnet, roberta, distilbert = load_models()
    test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()
    device = set_device()

    bert.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="bert-base-uncased", df =test_df)
    test_evaluate(test_df, test_data_loader, bert, device, pretrained_model="bert-base-uncased")
    del bert, test_data_loader

    xlnet.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="xlnet-base-cased", df=test_df)
    test_evaluate(test_df, test_data_loader, xlnet, device, pretrained_model="xlnet-base-cased")
    del xlnet, test_data_loader

    roberta.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="roberta-base", df=test_df)
    test_evaluate(test_df, test_data_loader, roberta, device, pretrained_model="roberta-base")
    del roberta, test_data_loader

    distilbert.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="distilbert-base-uncased", df=test_df)
    test_evaluate(test_df, test_data_loader, distilbert, device, pretrained_model="distilbert-base-uncased")
    del distilbert, test_data_loader

if __name__ == "__main__":
    evaluate_all_models()
