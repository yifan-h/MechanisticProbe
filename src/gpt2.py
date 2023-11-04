import os
import torch
import json
import random
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_constant_schedule_with_warmup, GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
import networkx as nx

from utils import data_generator, plot_loss, set_grad, avg


def evaluator(args, tokenizer, model, eval_data, eval_labels, batch_num, test=False, head_mask=None):
    if not test:
        rand_idx = random.choices([i for i in range(len(eval_labels))], k=1000)
        eval_data = [eval_data[idx] for idx in rand_idx]
        eval_labels = [eval_labels[idx] for idx in rand_idx]
    acc_list = []
    with torch.no_grad():
        for s in range(0, len(eval_labels), batch_num):
            tmp_data = eval_data[s:s+batch_num]
            tmp_label = eval_labels[s:s+batch_num]
            inputs = tokenizer(tmp_data, return_tensors="pt").to(args.device)
            if head_mask is not None:
                head_mask = head_mask.to(args.device)
                outputs = model(**inputs, labels=inputs["input_ids"], head_mask=head_mask).logits.softmax(-1)[:, -1, :]
            else:
                outputs = model(**inputs, labels=inputs["input_ids"]).logits.softmax(-1)[:, -1, :]
            predicts = torch.argmax(outputs, dim=1)
            labels = tokenizer(tmp_label, return_tensors="pt")["input_ids"].to(args.device)
            # if predict correctly, accuracy + 1
            labels = torch.squeeze(labels)
            for idx in range(labels.shape[0]):
                if labels[idx] == predicts[idx]:
                    acc_list.append(1.)
                else:
                    acc_list.append(0.)
    return avg(acc_list)


def finetune_gpt2(args):
    # prepare model
    # folder_name = args.model_path.split("/")[-1]
    folder_name = "training"
    epoch_num = args.epoch_num
    batch_num = 128  # 32 * 16 for acc, 128 * 2 for attn
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)

    # prepare data
    sdata_generator = data_generator(args, tokenizer)
    train_data, val_data, test_data = sdata_generator.return_data(data=True)
    train_label, val_label, test_label = sdata_generator.return_data(data=False)

    # training
    set_grad(model, args.tuning_param)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-3,)  # 2e-5 for 16/512, 1e-6 as default (+min00)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=500,)

    loss_fcn = torch.nn.CrossEntropyLoss()
    loss_list = []
    acc_list = [0]
    best_path = None
    with tqdm(total=int(epoch_num*len(train_label)/batch_num)) as t:
        for e in range(epoch_num):
            for s in range(0, len(train_label), batch_num):
                # updating
                tmp_data = train_data[s:s+batch_num]
                tmp_label = train_label[s:s+batch_num]
                inputs = tokenizer(tmp_data, return_tensors="pt").to(args.device)
                outputs = model(**inputs, labels=inputs["input_ids"]).logits.softmax(-1)[:, -1, :]
                labels = tokenizer(tmp_label, return_tensors="pt")["input_ids"].to(args.device)
                labels = torch.squeeze(labels)
                loss = loss_fcn(outputs, labels)
                loss_list.append(float(loss.item()))
                loss.backward()
                scheduler.step()
                if s % 2 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                # evaluation
                if (e*len(train_label)+s+1) % int(int(epoch_num*len(train_label)/batch_num) / 100) == 0:
                    acc_list.append(evaluator(args, tokenizer, model, val_data, val_label, batch_num))
                    # print(acc_list[-1])
                    if acc_list[-1] == max(acc_list):  # save best model
                        best_path = os.path.join(args.tmp_dir, folder_name, "gpt2_"+args.stask+"_"+args.tuning_param)
                        model.save_pretrained(best_path)
                # print results
                t.set_description("Training num: {:d}".format(s+1))
                t.set_postfix(loss=round(float(loss.item()), 6), acc=round(acc_list[-1], 6))
                t.update(1)

    # test
    acc_list.append(evaluator(args, tokenizer, model, test_data, test_label, batch_num, test=True))
    del model
    if best_path is not None:
        model = GPT2LMHeadModel.from_pretrained(best_path).to(args.device)
        acc_list.append(evaluator(args, tokenizer, model, test_data, test_label, batch_num, test=True))
        print("Test accuracy last and best (", args.stask, ") : ", acc_list[-2], "; ", acc_list[-1])
    else:
        print("Test accuracy  last (", args.stask, ") : ", acc_list[-1])

    # output results
    if not os.path.exists(os.path.join(args.tmp_dir, folder_name)):
        os.mkdir(os.path.join(args.tmp_dir, folder_name))
    results = {"acc": acc_list, "loss": loss_list}
    with open(os.path.join(args.tmp_dir, folder_name, "gpt2_"+args.stask+"_"+args.tuning_param+"_results.json"), "w") as f:
        f.write(json.dumps(results))
    plot_loss(loss_list, os.path.join(args.tmp_dir, folder_name, "gpt2_"+args.stask+"_"+args.tuning_param+"_loss.pdf"))
    plot_loss(acc_list, os.path.join(args.tmp_dir, folder_name, "gpt2_"+args.stask+"_"+args.tuning_param+"_acc.pdf"))
    return

