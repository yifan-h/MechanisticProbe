import os
import math
import torch
import json
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, get_constant_schedule_with_warmup
from tqdm import tqdm
from scipy.stats import entropy

import sys 
sys.path.append("..") 
from utils import data_generator, plot_entropy, plot_flow
from gpt2 import evaluator


def head_calculation(args, model, tokenizer, test_data, save_path="", title=""):
    '''
    # get head entropy map
    entropy_pos = np.zeros((12,12))
    entropy_size = np.zeros((12,12))
    # get attention list
    with torch.no_grad():
        for i in tqdm(range(len(test_data))):
            data = test_data[i]
            data_num = data.split(" ")
            data_sorted = sorted([int(d) for d in data_num])
            label_idx = [data_num.index(str(d)) for d in data_sorted[:16]]
            # get attention matrix
            inputs = tokenizer(data, return_tensors="pt").to(args.device)
            outputs = model(**inputs, output_attentions=True).attentions  # [1 x attn head x N x N]
            tmp_size, tmp_pos = [], []
            for attn in outputs:
                attn = attn[:,:,-1,:].cpu().numpy()
                attn_pos = attn  # [1 x attn head x N]
                tmp_pos.append(attn_pos)
                attn_size = [attn[:,:,l_idx:l_idx+1] for l_idx in label_idx]
                attn_size = np.concatenate(attn_size, axis=-1)
                tmp_size.append(attn_size)
            attn_pos = np.concatenate(tmp_pos, axis=0)  # [layer_num x attn head x N]
            attn_size = np.concatenate(tmp_size, axis=0)  # [layer_num x attn head x N]
            for l in range(attn_pos.shape[0]):
                for h in range(attn_pos.shape[1]):
                    entropy_pos[l, h] += entropy(attn_pos[l,h,:]) / math.log(attn_pos.shape[2])
                    entropy_size[l, h] += entropy(attn_size[l,h,:]) / math.log(attn_size.shape[2])
    entropy_pos = entropy_pos / len(test_data)  # [layer_num x attn head x N]
    entropy_size = entropy_size / len(test_data)  # [layer_num x attn head x N]
    xlabel = [str(i+1) for i in range(12)]
    ylabel1 = xlabel
    if len(save_path):
        plot_entropy(np.transpose(entropy_pos), save_path.replace(".pdf", "_pos_em.pdf"), xlabel=xlabel, ylabel1=ylabel1)
        plot_entropy(np.transpose(entropy_size), save_path.replace(".pdf", "_size_em.pdf"), xlabel=xlabel, ylabel1=ylabel1)
    '''
    # get attention list
    attn_pos_all, attn_size_all = None, None
    with torch.no_grad():
        for i in tqdm(range(len(test_data))):
            data = test_data[i]
            data_num = data.split(" ")
            data_sorted = sorted([int(d) for d in data_num])
            label_idx = [data_num.index(str(d)) for d in data_sorted[:16]]
            # get attention matrix
            inputs = tokenizer(data, return_tensors="pt").to(args.device)
            outputs = model(**inputs, output_attentions=True).attentions  # [1 x attn head x N x N]
            tmp_size, tmp_pos = [], []
            for attn in outputs:
                attn = attn[:,:,-1,:].cpu().numpy()
                attn_pos = attn  # [1 x attn head x N]
                tmp_pos.append(attn_pos)
                attn_size = [attn[:,:,l_idx:l_idx+1] for l_idx in label_idx]
                attn_size = np.concatenate(attn_size, axis=-1)
                tmp_size.append(attn_size)
            attn_pos = np.concatenate(tmp_pos, axis=0)  # [layer_num x attn head x N]
            if attn_pos_all is None:
                attn_pos_all = attn_pos
            else:
                attn_pos_all += attn_pos
            attn_size = np.concatenate(tmp_size, axis=0)  # [layer_num x attn head x N]
            if attn_size_all is None:
                attn_size_all = attn_size
            else:
                attn_size_all += attn_size
    attn_pos_all = attn_pos_all / len(test_data)  # [layer_num x attn head x N]
    attn_size_all = attn_size_all / len(test_data)  # [layer_num x attn head x N]
    # get head entropy map
    entropy_pos = np.zeros(attn_pos_all.shape[:2])
    entropy_size = np.zeros(attn_size_all.shape[:2])
    for l in range(attn_pos_all.shape[0]):
        for h in range(attn_pos_all.shape[1]):
            entropy_pos[l, h] = entropy(attn_pos_all[l,h,:]) / math.log(attn_pos_all.shape[2])
            entropy_size[l, h] = entropy(attn_size_all[l,h,:]) / math.log(attn_size_all.shape[2])
    xlabel = [str(i+1) for i in range(12)]
    ylabel1 = xlabel
    if len(save_path):
        plot_entropy(np.transpose(entropy_pos), save_path.replace(".pdf", "_pos.pdf"), xlabel=xlabel, ylabel1=ylabel1)
        plot_entropy(np.transpose(entropy_size), save_path.replace(".pdf", "_size.pdf"), xlabel=xlabel, ylabel1=ylabel1)
    return entropy_pos, entropy_size


def get_head_mask(e, p, drop="min"):  # [layer_num x attn head x N]
    entropy = e.copy()
    head_mask = torch.zeros(entropy.shape[0], entropy.shape[1]) + 1
    p_num = int(p / 100 * entropy.shape[0] * entropy.shape[1])
    p_count = 0
    while p_count <= p_num:
        # find the minimum entropy head
        if drop == "max":
            idxl, idxh = np.where(entropy==entropy.max())
            ridx = random.choice([i for i in range(len(idxl))])
            idxl = int(idxl[ridx])
            idxh = int(idxh[ridx])
            entropy[idxl, idxh] = 0.
        elif drop == "min":
            idxl, idxh = np.where(entropy==entropy.min())
            tmp_num = 1.
            ridx = random.choice([i for i in range(len(idxl))])
            idxl = int(idxl[ridx])
            idxh = int(idxh[ridx])
            entropy[idxl, idxh] = 1.
        else:
            idxl = random.choice([i for i in range(entropy.shape[0])])
            idxh = random.choice([i for i in range(entropy.shape[1])])
        # if sum(head_mask[idxl, :]) > 1:
        if head_mask[idxl, idxh] == 1:
            p_count += 1
            head_mask[idxl, idxh] = 0
    return head_mask


def acc_calculation(args, model, tokenizer, test_data, test_labels, save_path, title, entropy_pos, entropy_size):
    org_acc = evaluator(args, tokenizer, model, test_data, test_labels, 128, True)
    acc_list = []
    drop_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for d in drop_list:
        acc_list.append(evaluator(args, tokenizer, model, test_data, test_labels, 128, True, get_head_mask(entropy_pos, d, "rand")))
    print("The accuracy (drop randomly) is: ", [org_acc] + [round(acc, 6) for acc in acc_list])
    acc_list = []
    for d in drop_list:
        acc_list.append(evaluator(args, tokenizer, model, test_data, test_labels, 128, True, get_head_mask(entropy_pos, d, "min")))
    print("The accuracy (drop min pos entropy first) is: ", [org_acc] + [round(acc, 6) for acc in acc_list])
    acc_list = []
    for d in drop_list:
        acc_list.append(evaluator(args, tokenizer, model, test_data, test_labels, 128, True, get_head_mask(entropy_size, d, "min")))
    print("The accuracy (drop min size entropy first) is: ", [org_acc] + [round(acc, 6) for acc in acc_list])
    return


def pruning_analysis(args):
    folder_name = "prune"
    if not os.path.exists(os.path.join(args.tmp_dir, folder_name)):
        os.mkdir(os.path.join(args.tmp_dir, folder_name))
    # get all folders (GPT-2)
    for root, dirs, files in os.walk(os.path.join(args.tmp_dir, "training")):
        folders = dirs
        break
    folders = [os.path.join(args.tmp_dir, "training", f_path) for f_path in folders if f_path[:8]=="gpt2_min"]
    new_folders = []
    for i in range(len(folders)):
        if "attn" in folders[i]:
            continue
        elif "mlp" in folders[i]:
            continue
        elif "ln" in folders[i]:
            continue
        else:
            new_folders.append(folders[i])
    folders = new_folders
    model_label = [f_name[20:20+5] for f_name in folders]
    # set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(folders, model_label)
    for i in range(len(folders)):
        # prepare data
        args.stask = model_label[i][:5]
        sdata_generator = data_generator(args, tokenizer)
        _, _, test_data = sdata_generator.return_data(data=True)
        _, _, test_labels = sdata_generator.return_data(data=False)

        save_path = os.path.join(args.tmp_dir, folder_name, "pruning_analysis-all_"+model_label[i]+".pdf")
        title = model_label[i]
        model = AutoModelForCausalLM.from_pretrained(folders[i]).to(args.device)
        entropy_pos, entropy_size = head_calculation(args, model, tokenizer, test_data, save_path, title)
        acc_calculation(args, model, tokenizer, test_data, test_labels, save_path, title, entropy_pos, entropy_size)

    # visualize case study
    if not os.path.exists(os.path.join(args.tmp_dir, "gpt2_cs")):
        os.mkdir(os.path.join(args.tmp_dir, "gpt2_cs"))
    args.stask = "min01"
    sdata_generator = data_generator(args, tokenizer)
    _, _, test_data = sdata_generator.return_data(data=True)

    f_path = [f for f in folders if "min01" in f][0]
    model = AutoModelForCausalLM.from_pretrained(f_path).to(args.device)
    entropy_pos, entropy_size = head_calculation(args, model, tokenizer, test_data,)
    head_mask_pos = get_head_mask(entropy_pos, 40, "min").to(args.device)  # remove 40% position heads ([layer_num x attn head])
    attn_mean_all, pos_mean_all = None, None
    pos_00, pos_01 = 8, 12
    with torch.no_grad():
        for data in test_data:
            inputs = tokenizer(data, return_tensors="pt").to(args.device)
            outputs = model(**inputs, output_attentions=True, head_mask=head_mask_pos).attentions  # [1 x attn head x N x N]
            attn_pos = torch.cat(outputs, dim=0)  # [layer_num x attn head x N x N]
            mask_sum = torch.sum(head_mask_pos, dim=1).unsqueeze(-1).unsqueeze(-1)
            attn_pos = torch.sum(attn_pos, dim=1) / mask_sum  # [layer_num x N x N]
            attn_pos = attn_pos.cpu().numpy()
            if pos_mean_all is None:
                pos_mean_all = attn_pos
            else:
                pos_mean_all += attn_pos

            data_num = data.split(" ")
            if len(data_num) != len(set(data_num)): continue
            data_sorted = sorted([int(d) for d in data_num])
            data_min00, data_min01 = str(data_sorted[0]), str(data_sorted[1])
            data_drop_00, data_drop_01 = data_num[pos_00-1], data_num[pos_01-1]
            data_num[data_num.index(data_min00)] = data_drop_00
            data_num[data_num.index(data_min01)] = data_drop_01
            data_num[pos_00-1] = data_min00
            data_num[pos_01-1] = data_min01
            data = " ".join(data_num)

            inputs = tokenizer(data, return_tensors="pt").to(args.device)
            outputs = model(**inputs, output_attentions=True, head_mask=head_mask_pos).attentions  # [1 x attn head x N x N]
            attn = torch.cat(outputs, dim=0)  # [layer_num x attn head x N x N]
            mask_sum = torch.sum(head_mask_pos, dim=1).unsqueeze(-1).unsqueeze(-1)
            attn_mean = torch.sum(attn, dim=1) / mask_sum  # [layer_num x N x N]
            attn_mean = attn_mean.cpu().numpy()
            if attn_mean_all is None:
                attn_mean_all = attn_mean
            else:
                attn_mean_all += attn_mean
        attn_mean_all = attn_mean_all/len(test_data)
        pos_mean_all = pos_mean_all/len(test_data)
        xlabel=[""] + [str(i+1) for i in range(len(outputs))]
        ylabel = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
        ylabel.reverse()
        save_path = os.path.join(args.tmp_dir, "gpt2_cs", "cs_min01_all.pdf")
        plot_flow(pos_mean_all, save_path, "", xlabel=xlabel, ylabel=ylabel, cs=True)
        ylabel = ["1", "2", "3", "4", "5", "6", "7", "8 (leaf)", "9", "10", "11", "12 (root)", "13", "14", "15", "16"]
        ylabel.reverse()
        save_path = os.path.join(args.tmp_dir, "gpt2_cs", "cs_min01_"+str(pos_00)+"_"+str(pos_01)+".pdf")
        plot_flow(attn_mean_all, save_path, "", xlabel=xlabel, ylabel=ylabel, cs=True, pos1=pos_00, pos2=pos_01)

    return
