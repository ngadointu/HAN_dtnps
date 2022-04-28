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
    print("test_acc: " + str(test_acc))
    print("test_recall_pc: " + str(test_recall_pc))
    print("test_recall: " + str(test_recall))
    print("test_f1: " + str(test_f1))
    print("test_prec_pc: " + str(test_prec_pc))
    print("test_prec: " + str(test_prec))
    cm = confusion_matrix(test_ys, preds)
    real_tnps = 1.0 * (sum(cm[1]) - sum(cm[0])) / (sum(cm[0]) + sum(cm[1]) + sum(cm[2]))
    predicted_tnps = 1.0 * (sum(cm[:, 1]) - sum(cm[:, 0])) / (sum(cm[:, 0]) + sum(cm[:, 1]) + sum(cm[:, 2]))
    print("real tnps:")
    print(real_tnps)
    print("predicted tnps:")
    print(predicted_tnps)
    print("delta:")
    print(real_tnps - predicted_tnps)
    print(cm[:, 0])
    print(cm)


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


if __name__ == "__main__":

    args = parse_args()
    call_type = "chat_" if args.call_type == "chat" else ""
    
    bucket_name = 'nlp-ilexpo-prd'
    train_path = 'tnps/'+call_type+'transcript_metadata_'+args.bu+'_2020_tokenized/train/'
    val_path = 'tnps/'+call_type+'transcript_metadata_'+args.bu+'_2020_tokenized/val/'
    test_path = 'tnps/'+call_type+'transcript_metadata_'+args.bu+'_2020_tokenized/test/'
    tok_path = 'tnps/transcript_metadata_'+args.bu+'_2020_filtered/HANPreprocessor.joblib'
    s3 = boto3.client('s3')
    s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(bucket_name)

    log_dir = Path(args.log_dir)
    model_weights = log_dir / "weights"
    results_tab = os.path.join(args.log_dir, args.bu+"_results_2020.csv")
    paths = [log_dir, model_weights]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
            
#     ftrain, fvalid, ftest = "han_train.pkl", "han_valid.pkl", "han_test.pkl"
    tokf = "HANPreprocessor.joblib"
    model_name = (
        args.model
        + "_"
        + args.bu
        + "_utt_"
        + str(args.num_utt) 
        + "_lr_"
        + str(args.lr) 
        + "_num_classes_"
        + str(args.num_class)
        + "_wdc_"
        + str(args.weight_decay)
        + "_bsz_"
        + str(args.batch_size)
        + "_whd_"
        + str(args.word_hidden_dim)
        + "_shd_"
        + str(args.sent_hidden_dim)
        + "_emb_"
        + str(args.embed_dim)
        + "_semb_"
        + str(args.speaker_embed_dim)
        + "_drp_"
        + str(args.last_drop)
        + "_sch_"
        + str(args.lr_scheduler).lower()
        + "_cycl_"
        + (str(args.n_cycles) if args.lr_scheduler.lower() == "cycliclr" else "no")
        + "_lrp_"
        + (str(args.lr_patience) if args.lr_scheduler.lower() == "reducelronplateau" else "no")
        + "_weights_"
        + str(args.weights)
        + "_alpha_"
        + str(args.alpha)
        + "_gamma_"
        + str(args.gamma)
        + "_pre_"
        + ("no" if args.embedding_matrix is None else "yes")
        + "_with_weighted_sampler_"
        + "_all_2020"
    )
    print(model_name)
    
    if not args.test_only:
        train_dataset = HANDataset(bucket_name, train_path, "train")
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=n_cpus)

    eval_dataset = HANDataset(bucket_name, val_path, "dev")
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, num_workers=n_cpus)

    test_dataset = HANDataset(bucket_name, test_path, "test")
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=n_cpus)
    
#     key = prefix + tokf
    #tok = pickle.load(open(data_dir / tokf, "rb"))
    with tempfile.TemporaryFile() as fp:
        s3.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=tok_path)
        fp.seek(0)
        tok = joblib.load(fp)

    if args.embedding_matrix is not None:
        embedding_matrix = build_embeddings_matrix(tok.vocab, args.embedding_matrix, verbose=0)

    if args.model == "han":
        model = HierAttnNet(
            vocab_size=len(tok.vocab.stoi),
            maxlen_sent=tok.maxlen_sent,
            maxlen_doc=tok.maxlen_doc,
            word_hidden_dim=args.word_hidden_dim,
            sent_hidden_dim=args.sent_hidden_dim,
            padding_idx=args.padding_idx,
            embed_dim=args.embed_dim,
            weight_drop=args.weight_drop,
            embed_drop=args.embed_drop,
            locked_drop=args.locked_drop,
            last_drop=args.last_drop,
            embedding_matrix=args.embedding_matrix,
            num_class=args.num_class,
            speaker_embedding_dim=args.speaker_embed_dim,
        )
    if use_cuda:
        print("one gpu!")
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metric = CategoricalAccuracy()
    loss_function = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    best_epoch = 0

    if not args.test_only:
        training_steps = len(train_loader)
        step_size = training_steps * (args.n_epochs // (args.n_cycles * 2))
        if args.lr_scheduler.lower() == "reducelronplateau":
            # Since this scheduler runs within evaluation, and we evaluate every
            # eval_every epochs. Therefore the n_epochs before decreasing the lr
            # is lr_patience*eval_every (it we don't trigger early stop before)
            scheduler = ReduceLROnPlateau(optimizer, patience=args.lr_patience, factor=0.4)
        elif args.lr_scheduler.lower() == "cycliclr":
            scheduler = CyclicLR(
                optimizer,
                step_size_up=step_size,
                base_lr=args.lr,
                max_lr=args.lr * 10.0,
                cycle_momentum=False,
            )
        else:
            scheduler = None

        stop_step = 0
        best_loss = 1e6
        print("weights:")
        print(args.weights)
        for epoch in range(args.n_epochs):
            train_step(model, optimizer, train_loader, epoch, metric, scheduler=scheduler, 
                       loss_function=loss_function, weights=args.weights)
            if epoch % args.eval_every == (args.eval_every - 1):
                val_loss, _, _ = eval_step(model, eval_loader, metric)
                best_loss, stop_step, stop = early_stopping(
                    val_loss, best_loss, stop_step, args.patience
                )
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
            if stop:
                break
            if (stop_step == 0) & (args.save_results):
                best_epoch = epoch
                torch.save(model.state_dict(), model_weights / (model_name + ".pt"))

    if args.save_results:
        model.load_state_dict(torch.load(model_weights / (model_name + ".pt")))       

        val_loss, val_preds, val_ys = eval_step(model, eval_loader, metric, is_valid=True)
        val_probs = F.softmax(torch.cat(val_preds), 1).cpu()
        val_preds = val_probs.topk(1, 1)[1].cpu().numpy().squeeze(1)
        val_ys = torch.cat(val_ys).cpu().numpy()
        print("*** debugging ys start ***")
        print(val_preds)
        print("*** debugging ys ***")
        print(val_ys)
        print("*** debugging ys end ***")
        val_acc = accuracy_score(val_ys, val_preds)
        val_recall_pc = recall_score(val_ys, val_preds, average=None)
        val_recall = recall_score(val_ys, val_preds, average="weighted")
        val_f1 = f1_score(val_ys, val_preds, average="weighted")
        val_prec_pc = precision_score(val_ys, val_preds, average=None)
        val_prec = precision_score(val_ys, val_preds, average="weighted")
        print("val_acc: " + str(val_acc))
        print("val_recall_pc: " + str(val_recall_pc))
        print("val_recall: " + str(val_recall))
        print("val_f1: " + str(val_f1))
        print("val_prec_pc: " + str(val_prec_pc))
        print("val_prec: " + str(val_prec))
        cm = confusion_matrix(val_ys, val_preds)
        real_tnps = 1.0*(sum(cm[1]) - sum(cm[0])) / (sum(cm[0]) + sum(cm[1]) + sum(cm[2]))
        predicted_tnps = 1.0*(sum(cm[:,1]) - sum(cm[:,0])) / (sum(cm[:,0]) + sum(cm[:,1]) + sum(cm[:,2]))
        print("real tnps:")
        print(real_tnps)
        print("predicted tnps:")
        print(predicted_tnps)
        print("delta:")
        print(real_tnps-predicted_tnps)        
        print(cm[:,0])
        print(cm)

        test_loss, preds, test_ys = eval_step(model, test_loader, metric, is_valid=False)
        probs = F.softmax(torch.cat(preds), 1).cpu()
        preds = probs.topk(1, 1)[1].cpu().numpy().squeeze(1)
        test_ys = torch.cat(test_ys).cpu().numpy()
        print("*** debugging test ys start ***")
        print(preds)
        print("*** debugging test ys ***")
        print(test_ys)
        print("*** debugging test ys end ***")
        test_acc = accuracy_score(test_ys, preds)
        test_recall_pc = recall_score(test_ys, preds, average=None)
        test_recall = recall_score(test_ys, preds, average="weighted")
        test_f1 = f1_score(test_ys, preds, average="weighted")
        test_prec_pc = precision_score(test_ys, preds, average=None)
        test_prec = precision_score(test_ys, preds, average="weighted")
        print("test_acc: " + str(test_acc))
        print("test_recall_pc: " + str(test_recall_pc))
        print("test_recall: " + str(test_recall))
        print("test_f1: " + str(test_f1))
        print("test_prec_pc: " + str(test_prec_pc))
        print("test_prec: " + str(test_prec))
        cm = confusion_matrix(test_ys, preds)
        real_tnps = 1.0*(sum(cm[1]) - sum(cm[0])) / (sum(cm[0]) + sum(cm[1]) + sum(cm[2]))
        predicted_tnps = 1.0*(sum(cm[:,1]) - sum(cm[:,0])) / (sum(cm[:,0]) + sum(cm[:,1]) + sum(cm[:,2]))
        print("real tnps:")
        print(real_tnps)
        print("predicted tnps:")
        print(predicted_tnps)
        print("delta:")
        print(real_tnps-predicted_tnps)        
        print(cm[:,0])
        print(cm)

        cols = ["modelname", "loss", "acc", "f1", "prec", "recall", "best_epoch"]
        vals = [model_name, test_loss, test_acc, test_f1, test_prec, test_recall, best_epoch]
        print(cols)
        print(vals)
        if not args.test_only:
            if not os.path.isfile(results_tab):
                results_df = pd.DataFrame(columns=cols)
                experiment_df = pd.DataFrame(data=[vals], columns=cols)
                results_df = results_df.append(experiment_df, ignore_index=True, sort=False)
                results_df.to_csv(results_tab, index=False)
            else:
                results_df = pd.read_csv(results_tab)
                experiment_df = pd.DataFrame(data=[vals], columns=cols)
                results_df = results_df.append(experiment_df, ignore_index=True, sort=False)
                results_df.to_csv(results_tab, index=False)
