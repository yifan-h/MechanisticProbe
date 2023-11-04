import os
import torch
import json
import random
import numpy as np
from transformers import LlamaForCausalLM, LlamaConfig, get_constant_schedule_with_warmup, AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import networkx as nx

from utils import data_loader, plot_loss, avg, set_grad


def evaluator(args, tokenizer, model, eval_data, test=False, head_mask=None, train_data=[]):
    # evaluation or test
    if not test:
        rand_idx = random.choices([i for i in range(len(eval_data))], k=1000)
        eval_data = [eval_data[idx] for idx in rand_idx]
    acc_list = []
    # prompting
    with torch.no_grad():
        for s in range(len(eval_data)):
            # zs prompt or fs icl
            icl_examples = ""
            if len(train_data):
                if head_mask is not None:  # train_data: icl text
                    icl_examples = train_data[s]
                else:  # train_data: training data (need random selection)
                    icl = random.sample(train_data, k=args.icl_num)
                    for e in icl:
                        icl_examples += e["context"] + " " + e["question"] + ": True or False?" + str(e["answer"]) + "\n"
            # prompt construction: proofwriiter label: True: 5852; False: 7700
            tmp_data = icl_examples + eval_data[s]["context"] + " " + eval_data[s]["question"] + ": True or False?"
            if eval_data[s]["answer"]:
                tmp_label = 0
            else:
                tmp_label = 1
            # inference
            inputs = tokenizer(tmp_data, return_tensors="pt", add_special_tokens=False).to(args.device)
            outputs = model(**inputs, labels=inputs["input_ids"]).logits.softmax(-1)[:, -1, :][:,[5852, 7700]]
            predicts = torch.argmax(outputs)
            if predicts == tmp_label:
                acc_list.append(1.)
            else:
                acc_list.append(0.)
    return avg(acc_list)

