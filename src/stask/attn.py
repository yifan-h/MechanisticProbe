import os
import torch
import json
import random
import numpy as np
from scipy.special import softmax
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_constant_schedule_with_warmup
from tqdm import tqdm
import networkx as nx

import sys 
sys.path.append("..") 
from utils import data_generator, plot_sim


def attn_calculation(args, model, tokenizer, test_data, save_path, title, pos):
    # get attention list
    with torch.no_grad():
        dist_mat = None
        for i in tqdm(range(len(test_data))):
            data = test_data[i]
            data_num = data.split(" ")
            data_sorted = sorted([int(d) for d in data_num])
            if pos:
                label_idx = [i for i in range(len(data_sorted[:16]))]
            else:
                label_idx = [data_num.index(str(d)) for d in data_sorted[:16]]
            # get attention matrix
            inputs = tokenizer(data, return_tensors="pt").to(args.device)
            outputs = model(**inputs, output_attentions=True).attentions  # [batch_num x attn head x attn matrix]
            tmp_dist_mat = []
            for attn in outputs:
                attn = torch.squeeze(torch.mean(attn, dim=1))[-1,:].cpu().numpy()  # shape: [16]
                tmp_dist = [attn[l_idx] for l_idx in label_idx]
                tmp_dist_mat.append(tmp_dist)
            if dist_mat is None:
                dist_mat = np.array(tmp_dist_mat)
            else:
                dist_mat += np.array(tmp_dist_mat)
        dist_mat = np.transpose(dist_mat) / len(test_data)
        if pos:
            ylabel = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", \
                        "9th", "10th", "11th", "12th", "13th", "14th", "15th", "16th"]
        else:
            ylabel = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", \
                        "9th", "10th", "11th", "12th", "13th", "14th", "15th", "16th"]
        plot_sim(dist_mat, save_path, title, xlabel=[str(i+1) for i in range(len(outputs))], ylabel1=ylabel, pos=pos)
    return


def attn_analysis(args, pos=False):
    if pos:
        folder_name = "attn_pos"
    else:
        folder_name = "attn"
    if not os.path.exists(os.path.join(args.tmp_dir, folder_name)):
        os.mkdir(os.path.join(args.tmp_dir, folder_name))
    # get all folders (GPT-2)
    for root, dirs, files in os.walk(os.path.join(args.tmp_dir, "training")):
        folders = dirs
        break
    folders = [args.model_path] + [os.path.join(args.tmp_dir, "training", f_path) for f_path in folders if f_path[:8]=="gpt2_min"]
    model_label = ["org"] + [f_name[20:20+5] for f_name in folders[1:]]
    # set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # prepare data
    sdata_generator = data_generator(args, tokenizer)
    _, _, test_data = sdata_generator.return_data(data=True)
    print(folders, model_label)
    for i in range(len(folders)):
        if "attn" in folders[i]:
            save_path = os.path.join(args.tmp_dir, folder_name, "attn_analysis-attn_"+model_label[i]+".pdf")
        elif "mlp" in folders[i]:
            save_path = os.path.join(args.tmp_dir, folder_name, "attn_analysis-mlp_"+model_label[i]+".pdf")
        elif "ln" in folders[i]:
            save_path = os.path.join(args.tmp_dir, folder_name, "attn_analysis-ln_"+model_label[i]+".pdf")
        else:
            save_path = os.path.join(args.tmp_dir, folder_name, "attn_analysis-all_"+model_label[i]+".pdf")
        title = model_label[i]
        model = AutoModelForCausalLM.from_pretrained(folders[i]).to(args.device)
        attn_calculation(args, model, tokenizer, test_data, save_path, title, pos)
    return
