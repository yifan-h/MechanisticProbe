import os
import torch
import json
import random
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
from torch.optim import Adam
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
import networkx as nx

import sys 
sys.path.append("..") 
from utils import data_generator, avg, plot_sim


def label_generator(data, task_name, pred_type):
    data_num = data.split(" ")
    data_sorted = sorted([int(d) for d in data_num])
    label_idx = [data_sorted.index(int(d)) for d in data_num]
    if pred_type == "noise":
        label_tensor = torch.zeros(len(data_num), 1)
        for i in range(task_name+1):
            if i not in label_idx:  # repeat numbers here
                k = i
                while k not in label_idx:
                    k = k-1
                label_tensor[label_idx.index(k), 0] = 1
            else:
                label_tensor[label_idx.index(i), 0] = 1
        if task_name not in label_idx:  # repeat numbers here
                k = task_name
                while k not in label_idx:
                    k = k-1
                label_tensor[label_idx.index(k), 0] = 1
        else:
            label_tensor[label_idx.index(task_name), 0] = 1
        # return torch.flatten(label_tensor)
        return label_tensor, [i for i in range(label_tensor.shape[0])]
    else:
        label_tensor = torch.zeros(len(data_num), 2)
        for i in range(task_name+1):
            if i not in label_idx:  # repeat numbers here
                k = i
                while k not in label_idx:
                    k = k-1
                label_tensor[label_idx.index(k), 0] = 1
            else:
                label_tensor[label_idx.index(i), 0] = 1
        if task_name not in label_idx:  # repeat numbers here
                k = task_name
                while k not in label_idx:
                    k = k-1
                label_tensor[label_idx.index(k), 1] = 1
        else:
            label_tensor[label_idx.index(task_name), 1] = 1
        # return torch.flatten(label_tensor)
        sample_idx = []
        for i in range(label_tensor.shape[0]):
            if torch.sum(label_tensor[i, :]) != 0:
                sample_idx.append(i)
        return label_tensor, sample_idx


def knn_classifier(args, task_name, attn_list, test_data, pred_type="depth"):
    acc_list = []
    label_list = []
    new_attn_list = []

    for i in range(len(test_data)):
        label, sample_idx = label_generator(test_data[i], task_name, pred_type)
        label_list.append(label[sample_idx])
        new_attn_list.append(attn_list[i][sample_idx])
    attn_list = new_attn_list
    attn_all = torch.cat(attn_list, dim=0).numpy()
    label_all = torch.squeeze(torch.cat(label_list, dim=0)).numpy()

    neigh = KNeighborsClassifier(n_neighbors=8, weights='distance')
    neigh.fit(attn_all[:int(len(label_list)/2)], label_all[:int(len(label_list)/2)])
    pred = neigh.predict(attn_all[int(len(label_list)/2):])
    targt = label_all[int(len(label_list)/2):]
    acc_list.append(f1_score(targt, pred, average="macro"))

    neigh = KNeighborsClassifier(n_neighbors=8, weights='distance')
    neigh.fit(attn_all[int(len(label_list)/2):], label_all[int(len(label_list)/2):])
    pred = neigh.predict(attn_all[:int(len(label_list)/2)])
    targt = label_all[:int(len(label_list)/2)]
    acc_list.append(f1_score(targt, pred, average="macro"))
    return acc_list


def probe_calculation(args, model, tokenizer, test_data, model_name, layer_wise=False):
    attn_list = []
    # get attention list
    with torch.no_grad():
        for i in tqdm(range(len(test_data))):
            data = test_data[i]
            # get attention matrix
            inputs = tokenizer(data, return_tensors="pt").to(args.device)
            outputs = model(**inputs, output_attentions=True).attentions  # [batch_num x attn head x attn matrix]
            tmp_attn_list = []
            for j in range(len(outputs)):
                attn = torch.squeeze(outputs[j])[:,-1,:].cpu()  # shape: [12x16]
                tmp_attn_list.append(attn)
            tmp_attn_list = [torch.mean(a, dim=0, keepdim=True) for a in tmp_attn_list]
            attn = torch.cat(tmp_attn_list, dim=0)
            attn = torch.transpose(attn, 0, 1)  # shape: [16x12]
            attn = attn/torch.max(attn)
            attn_list.append(attn)
    if layer_wise:
        task_name = int(model_name[-2:])
        for l in range(attn_list[0].shape[1]):
            tmp_attn_list = [a[:, :(l+1)] for a in attn_list]
            f1_noise_list = avg(knn_classifier(args, task_name, tmp_attn_list, test_data, "noise"))
            f1_depth_list = avg(knn_classifier(args, task_name, tmp_attn_list, test_data, "depth"))
            print(model_name, "Layer: " + str(l) + " KNN classifier (noise, depth) F1-Macro: ", round(f1_noise_list, 6), round(f1_depth_list, 6))
    else:
        # train & test probe model
        if "pretrained" in model_name or "scratch" in model_name:
            f1_noise_list, f1_depth_list = [], []
            for tn in range(int(args.stask_len/2)):
                f1_noise = avg(knn_classifier(args, tn, attn_list, test_data, "noise"))
                f1_depth = avg(knn_classifier(args, tn, attn_list, test_data, "depth"))
                print(model_name, "k=", tn, " KNN classifier (noise, depth) F1-Macro: ", round(f1_noise, 6), round(f1_depth, 6))
                f1_noise_list.append(f1_noise)
                f1_depth_list.append(f1_depth)
        else:
            task_name = int(model_name[-2:])
            f1_noise = avg(knn_classifier(args, task_name, attn_list, test_data, "noise"))
            f1_depth = avg(knn_classifier(args, task_name, attn_list, test_data, "depth"))
            print(model_name, " KNN classifier (noise, depth) F1-Macro: ", round(f1_noise, 6), round(f1_depth, 6))
            f1_noise_list = f1_noise
            f1_depth_list = f1_depth
    return f1_noise_list, f1_depth_list


def probe_analysis(args):
    folder_name = "training"
    # folder_name = args.model_path.split("/")[-1]
    if not os.path.exists(os.path.join(args.tmp_dir, "probe")):
        os.mkdir(os.path.join(args.tmp_dir, "probe"))
    # get all folders (GPT-2)
    for root, dirs, files in os.walk(os.path.join(args.tmp_dir, folder_name)):
        folders = dirs
        break
    folders = ["scratch", args.model_path] + [os.path.join(args.tmp_dir, folder_name, f_path) for f_path in folders if f_path[:8]=="gpt2_min"]
    model_label = ["scratch", "pretrained"] + [f_name[12+len(folder_name):12+len(folder_name)+5] for f_name in folders[2:]]
    # set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # prepare data
    sdata_generator = data_generator(args, tokenizer)
    _, _, test_data = sdata_generator.return_data(data=True)
    test_data = random.sample(test_data, k=1024)
    print(folders, model_label)
    for i in range(len(folders)):
        # if i < 2: continue
        if folders[i] == "scratch":
            config = GPT2Config()
            model = GPT2LMHeadModel(config=config).to(args.device)
            f1_noise_scratch, f1_depth_scratch = probe_calculation(args, model, tokenizer, test_data, model_label[i])
        elif model_label[i] == "pretrained":
            model = AutoModelForCausalLM.from_pretrained(folders[i]).to(args.device)
            f1_noise_pretrained, f1_depth_pretrained = probe_calculation(args, model, tokenizer, test_data, model_label[i])
            for j in range(len(f1_noise_pretrained)):
                sp1 = (f1_noise_pretrained[j]-f1_noise_scratch[j]) / (1-f1_noise_scratch[j])
                sp2 = (f1_depth_pretrained[j]-f1_depth_scratch[j]) / (1-f1_depth_scratch[j])
                print("====>>>> The pretrained GPT-2 probing scores are: (k=", str(j), ")", round(sp1, 6), round(sp2, 6))
        else:
            model = AutoModelForCausalLM.from_pretrained(folders[i]).to(args.device)
            f1_noise, f1_depth = probe_calculation(args, model, tokenizer, test_data, model_label[i])
            label_k = int(model_label[i][3:])
            sp1 = (f1_noise-f1_noise_scratch[label_k]) / (1-f1_noise_scratch[label_k])
            sp2 = (f1_depth-f1_depth_scratch[label_k]) / (1-f1_depth_scratch[label_k])
            print("====>>>> The finetuned GPT-2 probing scores are: ", round(sp1, 6), round(sp2, 6))
    return


def probe_layer(args):
    folder_name = "training"
    for root, dirs, files in os.walk(os.path.join(args.tmp_dir, folder_name)):
        folders = dirs
        break
    folders = [os.path.join(args.tmp_dir, folder_name, f_path) for f_path in folders if f_path[:8]=="gpt2_min" and "all" in f_path]
    model_label = [f_name[20:20+5] for f_name in folders]
    # set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # prepare data
    sdata_generator = data_generator(args, tokenizer)
    _, _, test_data = sdata_generator.return_data(data=True)
    test_data = random.sample(test_data, k=1024)
    for i in range(len(folders)):
        model = AutoModelForCausalLM.from_pretrained(folders[i]).to(args.device)
        probe_calculation(args, model, tokenizer, test_data, model_label[i], layer_wise=True)
    return
