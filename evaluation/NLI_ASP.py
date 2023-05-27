from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, default_data_collator
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import csv
import pandas as pd
import os
os.chdir('/path/to/work_dir')
from torch.utils.data import DataLoader
from ASP_utils import *
from natsort import natsorted



if __name__ == '__main__':
    task = 'mnli'
    num_labels = 3
    unit_type = "si_single_peak"
    model_list = os.listdir("/path/to/model")
    for model_name in model_list:
        print("\n\n")
        print("%"*100)
        print(f'now turn to the {model_name}:')
        model_path = f'/path/to/model/{model_name}'
        config = AutoConfig.from_pretrained(model_path, num_labels=num_labels, finetuning_task=task)
        tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=True)
        model_mnli = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
        torch.cuda.set_device(0)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_mnli.to(device)
        data_path = f'/data'
        dir_list = os.listdir(data_path)
        dir_list.remove("ASP_fine-tuning")

        bs_size = 32
        scale_class = ['entailment','not_entailment']
        for test_dir in dir_list:


            save_path = f"result/NLI_model/{model_name}/{test_dir}"
            createDir(save_path)

            test_list = os.listdir(f'{data_path}/{test_dir}')
            test_list = natsorted(test_list)
            for test_type in test_list:

                print("="*100)
                
                dev_path = f'{data_path}/{test_dir}/{test_type}'
                nli_classes = ['entailment','not_entailment']
                dev_premise,dev_hypo, dev_label = read_scale_data(dev_path)

                print(f"test dataset: {test_type}, from {dev_path}")
                print(f"test model from: {model_path}")
                print(f"save to: {save_path}")
                dev_encoded = tokenizer(dev_premise, dev_hypo, truncation=True, padding='max_length', max_length=128)
                dev_dataset = ScaleDataset(dev_encoded,dev_label)
                data_loader = DataLoader(dev_dataset,batch_size=bs_size,shuffle=False)    

                acc = 0
                pred = []

                for i, batch in enumerate(data_loader):
                    with torch.no_grad():
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        if 'token_type_ids' in batch:
                            token_type_ids = batch['token_type_ids'].to(device)
                            outputs = model_mnli(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
                        else:
                            outputs = model_mnli(input_ids, attention_mask=attention_mask)
                        logits = outputs['logits']
                        output = logits.detach().cpu()
                        
                        poss = torch.softmax(output,dim=1).tolist()
                        pred_batch = np.argmax(poss,axis=1)

                        for idx in range(len(pred_batch)):
                            pred_index = pred_batch[idx]
                            pred_index = 1 if pred_index in [1,2] else 0
                            pred_class = scale_class[pred_index]


                            pred.append(pred_class)
                result = list(zip(dev_premise,dev_hypo,dev_label,pred))
                with open(f'{save_path}/{test_type}','w') as f:
                    for pair in result:
                        if pair[2] == -2:
                            text = [pair[0],pair[1],'discard','discard']
                        else:
                            text = [pair[0],pair[1],scale_class[pair[2]],pair[3]]
                        text = "\t".join(text)
                        f.write(text+"\n")