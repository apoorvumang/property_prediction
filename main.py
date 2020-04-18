import torch
from torch import nn
import numpy as np
from model import classifier
from tokenizer import tokenizer
import random
import argparse
import json
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--fold", type=str, default='0')
parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--embedding_dim", type=int, default=100)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_hidden_nodes", type=int, default=256)
parser.add_argument("--dataset", type=str, default='ecoli')
parser.add_argument('--validate_every', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=5000)
parser.add_argument('--write_test_predictions', type=int, default=1)
parser.add_argument('--outfile', type=str, default='test_predictions.csv')
args = parser.parse_args()

print(args)

def readData(fname):
    lines = []
    f = open(fname, 'r')
    for line in f:
        lines.append(line.strip().split(','))
    f.close()
    data = []
    for line in lines[1:]:
        if 'for_submission' in fname and 'train' in fname:
            data.append([line[1], int(line[2])])
        else:
            data.append([line[0], int(line[1])])
    return data

if args.dataset == 'ecoli':
    folder_name = 'train_cv/fold_' + args.fold + '/'
    pos_weight = torch.tensor((2335-120)/120, dtype=torch.float32)
    valid = readData(folder_name + 'dev.csv')
    test = readData(folder_name + 'test.csv')
    train = readData(folder_name + 'train.csv')
elif args.dataset == 'covid':
    folder_name = 'covid/'
    pos_weight = torch.tensor((290726-405)/290726, dtype=torch.float32)
    # pos_weight = torch.tensor(1.0, dtype=torch.float32)
    valid = readData(folder_name + 'dev.csv')
    test = readData(folder_name + 'test.csv')
    train = readData(folder_name + 'train.csv')
elif args.dataset == 'for_submission':
    folder_name = 'for_submission/'
    pos_weight = torch.tensor((2335-120)/120, dtype=torch.float32)
    # pos_weight = torch.tensor(1.0, dtype=torch.float32)
    valid = readData(folder_name + 'dev.csv')
    test = readData(folder_name + 'train.csv')
    train = readData(folder_name + 'train.csv')

tk = tokenizer(train + valid + test)

# def make_batch(data):
#     X = []
#     Y = []
#     lengths = []
#     weights = []
#     for d in data:
#         x, x_len = tk.tokenize(d[0])
#         y = d[1]
#         Y.append(float(y))
#         # X.append(strToLong(x, char2int, max_length))
#         X.append(x)
#         lengths.append(x_len)
#     X = np.stack(X, axis=0)
#     Y = np.array(Y)
#     return X, lengths, Y

def make_batch(data):
    # this weight is calculated as 120/2335 
    weight_of_0_class = 120/2335
    weight_of_1_class = 1 - weight_of_0_class
    # weight_of_0_class = 1
    # weight_of_1_class = 2335/120
    X = []
    Y = []
    lengths = []
    weights = []
    for d in data:
        x, x_len = tk.tokenize(d[0])
        y = d[1]
        if y == 1:
            weights.append(weight_of_1_class)
        else:
            weights.append(weight_of_0_class)
        Y.append(float(y))
        # X.append(strToLong(x, char2int, max_length))
        X.append(x)
        lengths.append(x_len)
    X = np.stack(X, axis=0)
    Y = np.array(Y)
    return X, lengths, Y, np.array(weights, dtype=np.float32)


#define hyperparameters
size_of_vocab = tk.vocab_len()
embedding_dim = args.embedding_dim
num_hidden_nodes = args.num_hidden_nodes
num_output_nodes = 1
num_layers = args.num_layers
bidirection = True
dropout = args.dropout
if args.num_layers == 1:
    dropout = 0.0

#instantiate the model
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, 
                   bidirectional = bidirection, dropout = dropout)

print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(f'The model has {count_parameters(model):,} trainable parameters')

    
def aucroc(y_scores, y_true):
    return roc_auc_score(y_true, y_scores)

def aucprc(y_scores, y_true):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(lr_recall, lr_precision)
    
device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')  

if args.lr != 0.0:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters())

#push to cuda if available
model = model.to(device)

model.train()
optimizer.zero_grad()

batch_size = args.batch_size
model.train()

best_valid = 0
best_test = 0

for i in range(args.num_epochs):
    random.shuffle(train)
    model.train()
    total_loss = 0
    for j in range(0, len(train), batch_size):
        optimizer.zero_grad()   
        text, text_lengths, y, weights = make_batch(train[j:batch_size+j])
        text = torch.from_numpy(text).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        predictions = model(text, text_lengths).squeeze()  
        # criterion = nn.BCEWithLogitsLoss(weight=weights,pos_weight=pos_weight)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = criterion.to(device)
        loss = criterion(predictions, y)  
        # criterion = nn.BCELoss(weight=weights)
        # criterion = nn.BCELoss()
        # criterion = criterion.to(device)
        # loss = criterion(torch.sigmoid(predictions), y) 
        loss.backward()       
        #update the weights
        optimizer.step()   
        total_loss += loss.item()
    #validate
    if i%args.validate_every == 0:
        model.eval()
        valid_auc = 0
        test_auc = 0
        test_batch_size = 128
        predictions_for_writing = []
        for dataset in ['valid', 'test']:
            if dataset == 'valid':
                d = valid
            else:
                d = test
            test_predictions = np.array([])
            test_targets = np.array([])
            for k in range(0, len(d), test_batch_size):
                text, text_lengths, y, _ = make_batch(d[k:k+test_batch_size])
                text = torch.from_numpy(text).to(device)
                #apply sigmoid for predictions
                predictions = torch.sigmoid(model(text, text_lengths).squeeze())
                predictions = predictions.cpu().detach().numpy()
                test_predictions = np.append(test_predictions, predictions)
                test_targets = np.append(test_targets, y)
            auc1 = aucroc(test_predictions, test_targets)
            auc2 = aucprc(test_predictions, test_targets)
            if dataset == 'valid':
                valid_auc = auc2
                predictions_for_writing = test_predictions
            else:
                test_auc = auc2
            print(str(i) + '\t', dataset + '\t', auc2, auc1, total_loss)
        if valid_auc > best_valid:
            best_valid = valid_auc
            best_test = test_auc
            if args.write_test_predictions != 0:
                print('Writing valid predictions')
                f = open(args.outfile, 'w')
                f.write('smiles,activity\n')
                for i in range(len(valid)):
                    smile_string = valid[i][0]
                    pred = str(round(predictions_for_writing[i], 4))
                    f.write(smile_string + ',' + pred + '\n')
                f.close()
                
        print('Best valid:', best_valid, 'Corr. Test:', best_test, 'Fold:', args.fold, 'Dataset:', args.dataset)
