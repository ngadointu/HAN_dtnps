import boto3
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import random

from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset
from utils.io import *


n_cpus = 1
use_cuda = torch.cuda.is_available()

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def eval_x(model, loader, metric, is_valid):
    test_loss, preds, test_ys = eval_step(model, test_loader, metric, is_valid=False)
    probs = F.softmax(torch.cat(preds), 1).cpu()
    preds = probs.topk(1, 1)[1].cpu().numpy().squeeze(1)
    test_ys = torch.cat(test_ys).cpu().numpy()
    test_acc = accuracy_score(test_ys, preds)
    test_recall_pc = recall_score(test_ys, preds, average=None)
    test_recall = recall_score(test_ys, preds, average="weighted")
    test_f1 = f1_score(test_ys, preds, average="weighted")
    test_prec_pc = precision_score(test_ys, preds, average=None)
    test_prec = precision_score(test_ys, preds, average="weighted")
    cm = confusion_matrix(test_ys, preds)
    real_tnps = 1.0 * (sum(cm[1]) - sum(cm[0])) / (sum(cm[0]) + sum(cm[1]) + sum(cm[2]))
    predicted_tnps = 1.0 * (sum(cm[:, 1]) - sum(cm[:, 0])) / (sum(cm[:, 0]) + sum(cm[:, 1]) + sum(cm[:, 2]))


def train_step(model, optimizer, train_loader, epoch, metric, scheduler=None, loss_function=F.cross_entropy, weights=[1,1,1]):
    model.train()
    metric.reset()
    train_steps = len(train_loader)
    running_loss = 0
    with trange(train_steps) as t:
        for batch_idx, (text, speaker, time, total_duration, ntt_count, ntt_duration, overtalk_count,
                        overtalk_duration, talk_duration, target) in zip(t, train_loader):
            t.set_description("epoch %i" % (epoch + 1))
            X = text.cuda() if use_cuda else text
            S = speaker.cuda() if use_cuda else speaker
            T = time.cuda() if use_cuda else time
            totald = total_duration.cuda() if use_cuda else total_duration
            nttc = ntt_count.cuda() if use_cuda else ntt_count
            nttd = ntt_duration.cuda() if use_cuda else ntt_duration
            otc = overtalk_count.cuda() if use_cuda else overtalk_count
            otd = overtalk_duration.cuda() if use_cuda else overtalk_duration
            talkd = talk_duration.cuda() if use_cuda else talk_duration
            y = target.cuda() if use_cuda else target

            optimizer.zero_grad()
            y_pred = model(X, S, T, totald, nttc, nttd, otc, otd, talkd)
            loss = F.cross_entropy(y_pred, y, weight=torch.Tensor(weights).cuda())
            #loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            if isinstance(scheduler, CyclicLR):
                scheduler.step()

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            acc, pcacc = metric(F.softmax(y_pred, dim=1), y)

            t.set_postfix(acc=acc, loss=avg_loss)


def eval_step(model, eval_loader, metric, is_valid=True):
    model.eval()
    metric.reset()
    eval_steps = len(eval_loader)
    running_loss = 0
    preds = []
    ys = []
    deltas = []
    with torch.no_grad():
        with trange(eval_steps) as t:
            for batch_idx, (text, speaker, time, total_duration, ntt_count, ntt_duration, overtalk_count,
                            overtalk_duration, talk_duration, target) in zip(t, eval_loader):
                if is_valid:
                    t.set_description("valid")
                else:
                    t.set_description("test")
                X = text.cuda() if use_cuda else text
                S = speaker.cuda() if use_cuda else speaker
                T = time.cuda() if use_cuda else time
                totald = total_duration.cuda() if use_cuda else total_duration
                nttc = ntt_count.cuda() if use_cuda else ntt_count
                nttd = ntt_duration.cuda() if use_cuda else ntt_duration
                otc = overtalk_count.cuda() if use_cuda else overtalk_count
                otd = overtalk_duration.cuda() if use_cuda else overtalk_duration
                talkd = talk_duration.cuda() if use_cuda else talk_duration
                y = target.cuda() if use_cuda else target
                y_pred = model(X, S, T, totald, nttc, nttd, otc, otd, talkd)
                loss = F.cross_entropy(y_pred, y)
                running_loss += loss.item()
                avg_loss = running_loss / (batch_idx + 1)
                acc, pcacc = metric(F.softmax(y_pred, dim=1), y)
                preds.append(y_pred)
                ys.append(y)
                samp_probs = F.softmax(y_pred, 1).cpu()
                samp_preds = samp_probs.topk(1, 1)[1].cpu().numpy().squeeze(1)
                cm = confusion_matrix(y.cpu(), samp_preds, labels=[0,1,2])
                real_tnps = 1.0*(sum(cm[1]) - sum(cm[0])) / (sum(cm[0]) + sum(cm[1]) + sum(cm[2]))
                predicted_tnps = 1.0*(sum(cm[:,1]) - sum(cm[:,0])) / (sum(cm[:,0]) + sum(cm[:,1]) + sum(cm[:,2]))
                deltas.append(abs(real_tnps-predicted_tnps))
                t.set_postfix(acc=acc, loss=avg_loss)
    print(np.mean(deltas), np.std(deltas))
    return avg_loss, preds, ys


def early_stopping(curr_value, best_value, stop_step, patience):
    if curr_value <= best_value:
        stop_step, best_value = 0, curr_value
    else:
        stop_step += 1
    if stop_step >= patience:
        print("Early stopping triggered. patience: {} log:{}".format(patience, best_value))
        stop = True
    else:
        stop = False
    return best_value, stop_step, stop


class HANDataset(Dataset):
    """
    The data is stored as multiple pkl file in s3.
    This dataset is reading at each step 3 files in random order, concat them together
    and outputs them as a chunk, until all the files are visited.
    This way, we hold in memory no more than 3 files at each training step ~7000 examples.
    """

    def __init__(self, bucket_name, path, data_type):
        self.path = path
        self.counter = 0
        self.df = pd.DataFrame()
        self.files = get_files_list(bucket_name, path)
        self.data_type = data_type

    def __len__(self):
        if self.data_type == "train":
            return 300000
        else:
            return 70000

    def __getitem__(self, idx):
        # Beginning of the chunk
        if idx == 0:
            # shuffle
            random.shuffle(self.files)
            self.gen = csv_gen(bucket_name, self.files)
        if idx == self.counter:
            # get next chunk
            self.df = next(self.gen)
            self.counter += len(self.df)
        rel_idx = idx - self.counter
        return df_to_torch(self.df, rel_idx)


class HANLocalDataset(Dataset):
    """
    When the files are available locally for training, use this dataset
    """

    def __init__(self, path, data_type):
        self.path = path
        self.counter = 0
        self.df = pd.DataFrame()
        self.files = get_files_list(path)
        self.data_type = data_type

    def __len__(self):
        if self.data_type == "train":
            return 100
        else:
            return 50

    def __getitem__(self, idx):
        # Beginning of the chunk
        if idx == 0:
            # shuffle
            random.shuffle(self.files)
            self.gen = csv_gen(bucket_name, self.files)
        if idx == self.counter:
            # get next chunk
            self.df = next(self.gen)
            self.counter += len(self.df)
        rel_idx = idx - self.counter
        return df_to_torch(self.df, rel_idx)


