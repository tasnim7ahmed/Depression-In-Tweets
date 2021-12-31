import argparse

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--max_length", default=128, type=int,  help='Maximum number of words in a sample')
    parser.add_argument("--train_batch_size", default=16, type=int,  help='Training batch size')
    parser.add_argument("--valid_batch_size", default=32, type=int,  help='Validation batch size')
    parser.add_argument("--test_batch_size", default=32, type=int,  help='Test batch size')
    parser.add_argument("--epochs", default=1, type=int,  help='Number of training epochs')
    parser.add_argument("-lr","--learning_rate", default=3e-5, type=float,  help='The learning rate to use')
    parser.add_argument("-wd","--weight_decay", default=1e-4, type=float,  help=' Decoupled weight decay to apply')
    parser.add_argument("--adamw_epsilon", default=1e-8, type=float,  help='Adam’s epsilon for numerical stability')
    parser.add_argument("--warmup_steps", default=0, type=int,  help='The number of steps for the warmup phase.')
    parser.add_argument("--classes", default=4, type=int, help='Number of output classes')
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout")
    parser.add_argument("--seed", type=int, default=7, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="gpu", help="Training device - cpu/gpu")
    parser.add_argument("--dataset", type=str, default="FGBC", help="Select Dataset - FGBC/Twitter")

    parser.add_argument("--pretrained_model", default="bert-base-uncased", type=str, help='Name of the pretrained model')
    parser.add_argument("--bert_hidden", default=768, type=int, help='Number of layer for Bert')
    parser.add_argument("--roberta_hidden", default=768, type=int, help='Number of layer for Roberta')
    parser.add_argument("--xlnet_hidden", default=768, type=int, help='Number of layer for XLNet')
    parser.add_argument("--distilbert_hidden", default=768, type=int, help='Number of layer for XLNet')
    parser.add_argument("--ensemble_type", type=str, default="max-voting", help="Ensemble type - max-voting or averaging")

    parser.add_argument("--dataset_path", default="../Dataset/", type=str, help='Path to dataset file')
    parser.add_argument("--model_path", default="../Models/", type=str, help='Save best model')
    parser.add_argument("--output_path", default="../Output/", type=str, help='Get predicted labels for test data')
    parser.add_argument("--figure_path", default="../Figures/", type=str, help='Directory for accuracy and loss plots')
    parser.add_argument("--split", default="no", type=str, help='If base file needs to be splitted into Train, Val, Test')

    return parser