import os
import torch
import torch.nn as nn
import json
import random
import numpy as np
from torch.optim import Adam
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import sys 
sys.path.append("..") 
from utils import data_loader, avg, plot_sim, fix_seed, label_parser
from llama import evaluator


def knn_classifier(args, task_type, attn_list, test_data, pred_type, depth, layerwise=-1):
    acc_list = []
    label_list = []
    new_attn_list = []
    label_len = 0
    if depth > 1: label_len = 1
    for i in range(len(test_data)):
        label, sample_idx = label_parser(test_data[i], label_len, pred_type)
        new_attn = attn_list[i]
        if layerwise < 0:
            label = label[sample_idx]
            new_attn = new_attn[sample_idx]
        label_list.append(label)
        new_attn_list.append(new_attn)
    # print("The number of test data: ", len(label_list), len(test_data))
    attn_list = new_attn_list
    attn_all = torch.cat(attn_list, dim=0).numpy()
    label_all = torch.squeeze(torch.cat(label_list, dim=0))
    if pred_type != "noise":
        if len(label_all.shape) == 1: label_all = label_all.unsqueeze(-1)
        label_all_new = torch.argmax(label_all, dim=1)
        if layerwise >= 0:
            tmp_idx_list = []
            for row in range(label_all.shape[0]):
                if torch.sum(label_all[row]) == 0:
                    label_all_new[row] = 0.
                    tmp_idx_list.append(row)
                else:
                    if label_all_new[row] == layerwise:
                        label_all_new[row] = 1.
                        tmp_idx_list.append(row)
            label_all_new = label_all_new[tmp_idx_list]
            attn_all = attn_all[tmp_idx_list]
        label_all = label_all_new
    label_all = label_all.numpy()
    score_list = []
    neigh = KNeighborsClassifier(n_neighbors=8, weights="distance", p=1)
    score_list.append(avg(cross_val_score(neigh, attn_all, label_all, cv=5, scoring='f1_macro')))
    return [max(score_list)]


def icl_analysis(args, model, tokenizer, test_dict, task_type, train_data=[], prune_list=[], corr=False):
    # get accuracy
    acc_dict = {}
    for depth, data in test_dict.items():
        acc_dict[depth] = evaluator(args, tokenizer, model, data, test=True, train_data=train_data)
        acc = acc_dict[depth]
    if not corr: print("    ICL testing accuracy: ", acc_dict)
    attn_dict = {}
    score_noise, score_depth = {}, {}
    for depth, test_data in test_dict.items():
        attn_list = []
        # get attention list
        with torch.no_grad():
            for i in range(len(test_data)):
                data = test_data[i]
                # construct icl examples
                icl_examples = ""
                if len(train_data):
                    icl = random.choices(train_data, k=args.icl_num)
                    for e in icl:
                        icl_examples += e["context"] + " " + e["question"] + ": True or False?" + str(e["answer"]) + "\n"
                # construct context
                context_text = ""
                index_list = []
                for k, v in data["triples"].items():
                    context_text += v
                    index_list.append(len(tokenizer.encode(v, add_special_tokens=False)))  # end token index
                for k, v in data["rules"].items():
                    context_text += v
                    index_list.append(len(tokenizer.encode(v, add_special_tokens=False)))  # end token index
                context_text = icl_examples + context_text + " " + data["question"] + " True or False?"
                sent_num = len(index_list)  # get number of sentences
                index_list = [len(tokenizer.encode(icl_examples, add_special_tokens=False))] + index_list
                index_list = [sum(index_list[:j+1]) for j in range(len(index_list))]  # get index
                index_list[0] += 1
                # get attention matrix
                inputs = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).to(args.device)
                outputs = model(**inputs, output_attentions=True).attentions  # [batch_num x attn head x attn matrix]
                tmp_attn_list = []
                for j in range(len(outputs)):
                    attn = torch.squeeze(outputs[j])[:, max(index_list):, :max(index_list)].cpu()  # shape: [attn head x question_num x context_num]
                    attn = torch.max(attn, dim=1)[0]  # shape: [attn head x context_num]
                    tmp_attn_list.append(attn)
                tmp_attn_list = [torch.mean(a, dim=0, keepdim=True) for a in tmp_attn_list]
                attn = torch.cat(tmp_attn_list, dim=0)
                attn = torch.transpose(attn, 0, 1)  # shape: [input_len x layer_num]
                # get attn flow across sentences
                new_attn = torch.zeros(sent_num, attn.shape[1], device=attn.device)
                for j in range(sent_num):
                    new_attn[j, :] = torch.mean(attn[index_list[j]:index_list[j+1], :], 0)
                if len(prune_list):
                    layer_list = [i for i in range(new_attn.shape[1]) if i not in prune_list]
                    new_attn = new_attn[:, layer_list]
                attn_list.append(new_attn)
                del new_attn
        # train & test probe model
        f1_noise = avg(knn_classifier(args, task_type, attn_list, test_data, "noise", int(depth)))
        f1_depth = avg(knn_classifier(args, task_type, attn_list, test_data, "depth", int(depth)))
        if corr:
            return acc, f1_noise, f1_depth
        print(task_type, "m = ", depth, " KNN Classifier F1-Macro (noise, depth): ", round(f1_noise, 6) , round(f1_depth, 6))
    return score_noise, score_depth


def ft_analysis(args, model, tokenizer, test_dict, task_type, prune_list=[], corr=False):
    attn_dict = {}
    score_noise, score_depth = {}, {}
    for depth, test_data in test_dict.items():
        attn_list = []
        # get attention list
        with torch.no_grad():
            for i in range(len(test_data)):
                data = test_data[i]
                # construct context
                context_text = ""
                index_list = []
                for k, v in data["triples"].items():
                    context_text += v
                    index_list.append(len(tokenizer.encode(v, add_special_tokens=False)))  # end token index
                for k, v in data["rules"].items():
                    context_text += v
                    index_list.append(len(tokenizer.encode(v, add_special_tokens=False)))  # end token index
                context_text = context_text + " " + data["question"] + " True or False?"
                sent_num = len(index_list)  # get number of sentences
                index_list = [0] + index_list
                index_list = [sum(index_list[:j+1]) for j in range(len(index_list))]  # get index
                index_list[0] += 1
                # get attention matrix
                inputs = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).to(args.device)
                outputs = model(**inputs, output_attentions=True).attentions  # [batch_num x attn head x attn matrix]
                tmp_attn_list = []
                for j in range(len(outputs)):
                    # flows to question community 
                    attn = torch.squeeze(outputs[j])[:, max(index_list):, :max(index_list)].cpu()  # shape: [attn head x question_num x context_num]
                    attn = torch.mean(attn, dim=1)  # shape: [attn head x context_num]
                    tmp_attn_list.append(attn)
                tmp_attn_list = [torch.mean(a, dim=0, keepdim=True) for a in tmp_attn_list]
                attn = torch.cat(tmp_attn_list, dim=0)
                attn = torch.transpose(attn, 0, 1)  # shape: [token_num x layer_num]
                # get attn flow across sentences
                new_attn = torch.zeros(sent_num, attn.shape[1], device=attn.device)
                for j in range(sent_num):
                    new_attn[j, :] = torch.mean(attn[index_list[j]:index_list[j+1], :], 0)
                    # new_attn[j, :] = torch.max(attn[index_list[j]:index_list[j+1], :], 0)[0]
                if len(prune_list):
                    layer_list = [i for i in range(new_attn.shape[1]) if i not in prune_list]
                    new_attn = new_attn[:, layer_list]
                attn_list.append(new_attn)
                del new_attn
        # train & test probe model
        f1_noise = avg(knn_classifier(args, task_type, attn_list, test_data, "noise", int(depth)))
        f1_depth = avg(knn_classifier(args, task_type, attn_list, test_data, "depth", int(depth)))
        if corr:
            return f1_noise, f1_depth
        print(task_type, "m = ", depth, " KNN Classifier F1-Macro (noise, depth): ", round(f1_noise, 6) , round(f1_depth, 6))
    return score_noise, score_depth


def proofwriter_probe_analysis(args):
    fix_seed(args.random_seed)
    if not os.path.exists(os.path.join(args.tmp_dir, "proofwriter_probe")):
        os.mkdir(os.path.join(args.tmp_dir, "proofwriter_probe"))
    # prepare data
    rdata_loader = data_loader(args)
    train_dict, dev_dict, test_dict = rdata_loader.return_data()
    train_data = []
    for k, v in train_dict.items():
        train_data += v
    # remove multi-choice samples
    test_dict = {depth: [d for d in test_dict[depth]+dev_dict[depth]+train_dict[depth] if "OR" not in d["proof"]] for depth, _ in test_dict.items() if int(depth) <= 1}
    test_dict_sample = {1: [], 2:[], 4:[], 8:[], 12:[], 16:[], 20:[], 24:[]}
    data_num = 1024
    for depth, data in test_dict.items():
        random.shuffle(data)
        for d in data:
            label, sample_idx = label_parser(d, depth, "depth")
            label = label[sample_idx]
            if torch.sum(label) >= label.shape[1]:
                if int(depth) == 1:
                    tmp_count = 0
                    tmp_count += len(d["triples"])
                    tmp_count += len(d["rules"])
                    if tmp_count in test_dict_sample and len(test_dict_sample[tmp_count]) < data_num:
                        test_dict_sample[tmp_count].append(d)
                elif int(depth) == 0 and len(test_dict_sample[1]) < data_num:
                    test_dict_sample[1].append(d)
    test_dict = test_dict_sample
    print("Sampled data for probing (all depth=1, key=m): ", {k:len(v) for k, v in test_dict_sample.items()})
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    # analysis random baselines
    config=LlamaConfig()
    model = LlamaForCausalLM(config).to(args.device)
    score_noise_scratch, score_depth_scratch = ft_analysis(args, model, tokenizer, test_dict, "scratch")
    del model
    # analysis ICL setting
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(args.device)
    prune_list = [31, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 18, 17]
    score_noise_icl, score_depth_icl = icl_analysis(args, model, tokenizer, test_dict, "in-context learning", train_data, prune_list=prune_list)
    return


def corrput_func(args, model, tokenizer, test_dict_sample, train_data, ctype=""):
    test_data = test_dict_sample[4]
    new_test_data = []
    for data in test_data:
        new_data = {k: v for k, v in data.items()}
        tmp_context = data["context"].split(". ")
        count = 10
        tmp_idx = 0
        while count < 10 or tmp_context[tmp_idx] in data['proof']:
            tmp_idx = random.choice([i for i in range(len(tmp_context))])
        tmp_context[tmp_idx] = "That " + tmp_context[tmp_idx] + " is false"
        tmp_context = ". ".join(tmp_context)
        new_data["context"] = tmp_context
        new_test_data.append(new_data)
    # get accuracy
    test_dict = {4: new_test_data}
    acc_dict = {}
    for depth, data in test_dict.items():
        acc_dict[depth] = evaluator(args, tokenizer, model, data, test=True, train_data=train_data)
        acc = acc_dict[depth]
    return acc


def proofwriter_corrupt_analysis(args):
    fix_seed(args.random_seed)
    if not os.path.exists(os.path.join(args.tmp_dir, "proofwriter_probe")):
        os.mkdir(os.path.join(args.tmp_dir, "proofwriter_probe"))

    # prepare data
    rdata_loader = data_loader(args)
    train_dict, dev_dict, test_dict = rdata_loader.return_data()
    train_data = []
    for k, v in train_dict.items():
        train_data += v
    # remove multi-choice samples
    test_dict = {depth: [d for d in test_dict[depth]+dev_dict[depth]+train_dict[depth] if "OR" not in d["proof"]] for depth, _ in test_dict.items() if int(depth) == 1}
    test_dict_sample = {4: random.sample(v, k=128) for depth, v in test_dict.items()}
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

    # get scratch
    config=LlamaConfig()
    model = LlamaForCausalLM(config).to(args.device)
    f1_noise_scratch, f1_depth_scratch = ft_analysis(args, model, tokenizer, test_dict_sample, "scratch", corr=True)
    del model

    # analysis ICL setting
    prune_list = [31, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 18, 17]
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(args.device)
    acc_list, sp1_list, sp2_list = [], [], []
    corrput_shuffle, corrput_noise, corrput_counter = [], [], []
    for i in tqdm(range(1024)):
        test_dict_sample = {4: random.sample(v, k=random.randint(64, 128)) for depth, v in test_dict.items()}
        acc, f1_noise, f1_depth = icl_analysis(args, model, tokenizer, test_dict_sample, "in-context learning", train_data, prune_list=prune_list, corr=True)
        acc_list.append(acc)
        sp1_list.append((f1_noise-f1_noise_scratch)/(1-f1_noise_scratch))
        sp2_list.append((f1_depth-f1_depth_scratch)/(1-f1_depth_scratch))
        # corruption
        acc_shuffle = 0.
        acc_noise = 0.
        acc_counter = corrput_func(args, model, tokenizer, test_dict_sample, train_data, "counter")
        corrput_shuffle.append(acc_shuffle-acc)
        corrput_noise.append(acc_noise-acc)
        corrput_counter.append(acc_counter-acc)
    del model
    results = {"acc": acc_list, "sp1": sp1_list, "sp2": sp2_list, "corrupt_shuffle": corrput_shuffle, "corrput_noise": corrput_noise, "corrput_counter": corrput_counter}
    save_path = os.path.join(args.tmp_dir, "proofwriter_probe", "corrput_results.json")
    with open(save_path, "w") as f:
        f.write(json.dumps(results))
    # print results
    bin_num = 8
    bin_len = (max(sp2_list) - min(sp2_list)) / bin_num
    bins_shuf, bins_noise, bins_count, bins_acc = [], [], [], []
    for b in range(bin_num):
        bin_lower = min(sp2_list) + bin_len*b
        bin_upper = min(sp2_list) + bin_len*(b+1)
        tmp_shuf, tmp_noise, tmp_count, tmp_acc = [], [], [], []
        for j in range(len(sp2_list)):
            if sp2_list[j] >= bin_lower and sp2_list[j] < bin_upper:
                tmp_shuf.append(corrput_shuffle[j])
                tmp_noise.append(corrput_noise[j])
                tmp_count.append(corrput_counter[j])
                tmp_acc.append(acc_list[j])
        bins_shuf.append(avg(tmp_shuf))
        bins_noise.append(avg(tmp_noise))
        bins_count.append(avg(tmp_count))
        bins_acc.append(avg(tmp_acc))
    print("final results: ", bins_shuf, bins_noise, bins_count, bins_acc, min(sp2_list), max(sp2_list))
    return
