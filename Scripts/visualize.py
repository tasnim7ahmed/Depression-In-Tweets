import matplotlib.pyplot as plt

from common import get_parser

parser = get_parser()
args = parser.parse_args()

def save_acc_curves(history):
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f"{args.figure_path}{args.pretrained_model}---acc---.pdf")
    plt.clf()

def save_loss_curves(history):
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Training and Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f"{args.figure_path}{args.pretrained_model}---loss---.pdf")
    plt.clf()