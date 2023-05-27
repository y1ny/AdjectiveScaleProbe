import csv
import re
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import logging
import torch
import pickle
import os
import math
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
def merge_dict(dic_a,dic_b):
    result_dic = {}
    for k, v in dic_a.items():
        for m, n in dic_b.items():
            if k == m:
                result_dic[k] = []
                result_dic[k].append(dic_a[k])
                result_dic[k].append(dic_b[k])
                dic_a[k] = result_dic[k]
                dic_b[k] = result_dic[k]
            else:
                result_dic[k] = dic_a[k]
                result_dic[m] = dic_b[m]
    return result_dic
def read_word(path):
    with open(path,'r') as f:
        word = []
        rows = f.readlines()
        for row in rows:
            row = row.strip()
            word.append(row)
    return word
def read_scale_data(path,p_id=0,h_id=1, l_id=2):
    with open(path,'r') as f:
        next(f)
        rows = f.readlines()
        premise = []
        hypothesis = []
        labels = []
        for row in rows:
            row = row.strip()
            text = row.split('\t')
            premise.append(text[p_id])
            hypothesis.append(text[h_id])
            if text[l_id] == 'entailment':
                label = 0
            elif text[l_id] == 'not_entailment':
                label = 1
            elif text[l_id] == 'discard':
                label = -2
            else:
                logging.warning(f'No label find in {row}')
                continue
            labels.append(label)
    return premise,hypothesis,labels
class ScaleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=[]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels !=[]:
            if isinstance(self.labels[idx],str):
                item['labels'] = self.labels[idx]
            else:
                item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class ZeroshotDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=[]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels !=[]:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
def flat_accuracy(preds, labels):
    
    """A function for calculating accuracy scores"""
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

def list2csv(data,path):
    with open(path,'w',newline='') as out:
        for row in data:
            out.write("\t".join(row)+'\n')
    return 


def save_pkl(path,data):
    with open(path,'wb') as f:
        pickle.dump(data,f)
def read_pkl(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data
def createDir(filePath):
    if os.path.exists(filePath):
        return
    else:
        try:
            os.mkdir(filePath)
        except Exception as e:
            os.makedirs(filePath)

def convert_inputs_to_sentences(x):
    if isinstance(x, str):
        x = x.split(" ")
    last_outer_idx = 0
    split_ids = [-1]
    stop_words = {".", "?", "!", "。", "？", "！"}
    stop_words_outer = {",", "，", ";", "；"}
    sentence_length = 128  

    for i, w in enumerate(x):
        if w in stop_words_outer:
            last_outer_idx = i
        if i - split_ids[-1] > sentence_length:
            if last_outer_idx == split_ids[-1]:
                raise ValueError(
                    f"Sentence `{''.join(x[last_outer_idx: i + 1])}` is longer than `sentence_length (curr={sentence_length})`, please set it larger.")
            split_ids.append(last_outer_idx)
        elif w in stop_words:
            split_ids.append(i)
    if split_ids[-1] != len(x) - 1:
        split_ids.append(len(x) - 1)

    sentences = list()
    for start, end in zip(split_ids[:-1], split_ids[1:]):
        sentences.append(x[start + 1: end + 1])
    return sentences

def ppl_score(model, tokenizer, sentences, batch_size=1, verbose=False, *args, **kwargs):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    mask_id = int(tokenizer.convert_tokens_to_ids("<mask>"))
    all_probability = list()
    all_words = list()
    sentences = convert_inputs_to_sentences(sentences)
    for idx, sentence in enumerate(sentences):
        inputs = tokenizer(" ".join(sentence), return_tensors="pt")
        if 'token_type_ids' in inputs.keys():
            input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
                "attention_mask"]
        else:
            input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]            
        origin_ids = input_ids[0][1: -1]
        length = input_ids.shape[-1] - 2
        batch_indice = list()
        for i in range(length // batch_size): 
            batch_indice.append([i * batch_size, (i + 1) * batch_size])
        if length % batch_size != 0:
            batch_indice.append([batch_size * (length // batch_size), length])
        for start, end in batch_indice:
            with torch.no_grad():
                ids_list = list()
                for i in range(start, end):
                    tmp = input_ids.clone()
                    tmp[0][i + 1] = mask_id 
                    ids_list.append(tmp)
                new_input_ids = torch.cat(ids_list, dim=0) 
                new_attention_mask = attention_mask.expand(end - start, length + 2) 
                if 'token_type_ids' in inputs.keys():
                    new_token_type_ids = token_type_ids.expand(end - start, length + 2)
                    inputs = {
                        'input_ids': new_input_ids.to(device),
                        'token_type_ids': new_token_type_ids.to(device),
                        'attention_mask': new_attention_mask.to(device)
                    }
                else:
                    inputs = {
                        'input_ids': new_input_ids.to(device),
                        'attention_mask': new_attention_mask.to(device)
                    }
                outputs = model(**inputs).logits
                outputs = torch.softmax(outputs, dim=-1).detach().cpu().numpy()
                probability = [outputs[i][start + i + 1][ids] for i, ids in enumerate(origin_ids[start: end])]
                all_probability += probability
                all_words += tokenizer.convert_ids_to_tokens(origin_ids[start: end])

    if len(all_probability) == 0:
        l_score = 0
    else:
        l_score = sum([math.log(p, 2) for p in all_probability]) / len(all_probability)

    if verbose:
        words = list()
        for s in sentences:
            words += s
        for word, prob in zip(all_words, all_probability):
            print(f"{word} | {prob:.8f}")
        print(f"l score: {l_score:.8f}")
    ppl = math.pow(2, -1 * l_score)
    return ppl

def possigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))
def negsigmoid(x, a, b):
    return 1 - (1.0 / (1.0 + np.exp(-a*(x-b))))
def compute_metrics(pred_scale,label_scale,metric,n_pair, varible, polarity,is_weighted=False):

    if metric == "accuracy":
        if varible :
            xdata = np.arange(0.6,1.4,0.02)
            if polarity == 'pos':
                popt, pcov = curve_fit(possigmoid, xdata, label_scale, method='dogbox',)
                fit_y = possigmoid(xdata,*popt)
            elif polarity == 'neg':
                popt, pcov = curve_fit(negsigmoid, xdata, label_scale, method='dogbox',)
                fit_y = negsigmoid(xdata,*popt)


        else:
            fit_y = label_scale
        
        pred_dist = pred_scale

        label_binary = np.where(fit_y>=0.5,1,0)
        acc =0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        discard_num = 0
        for weight, l, p_all in zip(label_scale,label_binary,pred_dist):
            if p_all <0:
                discard_num += 1
                continue
            if is_weighted:
                acc += weight*p_all
                acc += (1-weight)*(n_pair-p_all)

            else:
                if l ==1:
                    tp += p_all
                    fp += (n_pair - p_all)
                    acc += (p_all)
                elif l==0:
                    tn += (n_pair - p_all)
                    fn += p_all
                    acc += (n_pair - p_all)

        result = acc/(n_pair*(len(label_scale)-discard_num))  
    elif metric == "mse":
        pred_scale /= n_pair
        result = mean_squared_error(label_scale,pred_scale)
    elif metric == 'correlation':
        pred_scale_p = []
        label_scale_p = []
        for p,l in zip(pred_scale,label_scale):
            if p>=0:
                pred_scale_p.append(p)
                label_scale_p.append(l)
        pred_scale_p = np.array(pred_scale_p).astype(float)
        pred_scale_p /= n_pair
        result = pearsonr(pred_scale_p,label_scale_p)[0]

    return result