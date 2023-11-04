import os
import re
import json
import random
import numpy as np
import networkx as nx
from networkx.algorithms.dag import dag_longest_path
import torch
from transformers import set_seed
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.manifold import SpectralEmbedding, Isomap, LocallyLinearEmbedding, MDS, TSNE 

from proofparser import get_proof_graph


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # Transformers
    # set_seed(seed)
    return


def set_grad(model, tuning_params="all"):
    if tuning_params == "attn":
        for name, param in model.named_parameters():
            if "attn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif tuning_params == "mlp":
        for name, param in model.named_parameters():
            if "mlp" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif tuning_params == "ln":
        for name, param in model.named_parameters():
            if "ln" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif tuning_params == "lora":
        from transformers.adapters import LoRAConfig
        config = LoRAConfig(r=8, alpha=16)
        model.add_adapter("lora_adapter", config=config)
        model.train_adapter("lora_adapter")
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True
    return


def avg(l):
    if len(l):
        return sum(l)/len(l)
    else:
        return 0


def label_parser(data, depth, pred_type="noise", corrput=""):
    depth = int(depth)
    # get context_key
    context_key = []
    for k, v in data["triples"].items():
        context_key.append(k)
    for k, v in data["rules"].items():
        context_key.append(k)
    # parse reasoning tree
    proof_text = data["proof"]
    nodes, edges = get_proof_graph(proof_text)
    # remove loops and repeat samples
    edges_noloop = []
    for src, dst in edges:
        if (dst, src) not in edges_noloop:
            edges_noloop.append((src, dst))
    edges = edges_noloop[:int(depth)]
    nodes = list(set(nodes))
    # random and shorcut
    if len(corrput) and pred_type != "noise":
        # print(len(edges), edges)
        if corrput == "random":
            random.shuffle(edges)
            if random.random() > 0.5: edges[0] = (edges[0][1], edges[0][0])
            if random.random() > 0.5: edges[1] = (edges[1][1], edges[1][0])
        elif corrput == "shortcut" or corrput == "231":
            edges = [edges[1], edges[0]]
        else:
            corrput_int = [int(corrput[s]) for s in range(len(corrput))]
            triples = [edges[0][0], edges[0][1], edges[1][1]]
            triples = [triples[corrput_int[i]-1] for i in range(len(triples))]
            edges = [(triples[0], triples[1]), (triples[1], triples[2])]
    # construct graph
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n)
    for src, dst in edges:
        g.add_edge(src, dst)

    if pred_type == "noise":
        label_tensor = torch.zeros(len(context_key), 1)
        for n in nodes:
            label_tensor[context_key.index(n), 0] = 1.
        return label_tensor, [i for i in range(len(context_key))]
    else:  # pred_type == "depth"
        # find root node
        root_nodes = [n for n in nodes]
        for src, dst in edges:
            if src in root_nodes:
                root_nodes.remove(src)
        if len(root_nodes) < 1:  # multiple root nodes / root node in loop
            # print("Warning! no root nodes (there is loop!): ", depth, nodes, edges, root_nodes)
            lp = []
            for n1 in g.nodes():
                for n2 in g.nodes():
                    all_paths = nx.all_simple_paths(g, n1, n2)
                    for p in all_paths:
                        if len(p) > len(lp): lp=p
            root_nodes = [lp[-1]]
        # get all node depth
        depth_dict = {}
        for n in nodes:
            max_len = 0
            for root in root_nodes:
                for l in nx.all_simple_paths(g, n, root):
                    max_len = max(len(l)-1, max_len)
            depth_dict[n] = max_len # nx.dag_longest_path_length(g, root)
        # add tensor
        label_tensor = torch.zeros(len(context_key), depth+1)
        # for all nodes
        for n in nodes:
            x_idx = context_key.index(n)
            y_idx = depth - depth_dict[n]
            label_tensor[x_idx, y_idx] = 1.
        '''
        # if there is undirected loop, actually there is none of loops
        for src, dst in edges:
            # src node
            if 1. not in label_tensor[context_key.index(src)]:
                label_tensor[context_key.index(src), depth-depth_dict[src]] = 1.
            if depth_dict[src] > 0:  # when root node has a loop
                if 1. not in label_tensor[context_key.index(dst)]:
                    label_tensor[context_key.index(dst), depth-depth_dict[src]+1] = 1.
            # dst node
            if 1. not in label_tensor[context_key.index(dst)]:
                label_tensor[context_key.index(dst), depth-depth_dict[dst]] = 1.
            if depth_dict[dst] < depth:  # when leaf node has a loop
                if 1. not in label_tensor[context_key.index(src)]:
                    label_tensor[context_key.index(src), depth-depth_dict[dst]-1] = 1.
        '''
        # check label
        sample_idx = [context_key.index(n) for n in nodes]
        '''
        if depth >= 1:
            test_label_tensor = label_tensor[sample_idx]
            if torch.sum(test_label_tensor) != test_label_tensor.shape[0]:
                print("Error label: mismatching number: ", test_label_tensor, sample_idx, context_key, nodes, edges)
            if torch.sum(test_label_tensor) < test_label_tensor.shape[1]:
                print("Error label: missing number: ", test_label_tensor, sample_idx, context_key, nodes, edges)
        '''
        return label_tensor, sample_idx


def plot_loss(loss_list, save_path):
    if len(loss_list) == 0: return
    while len(loss_list) >= 10000:
        new_loss_list = []
        for i in range(0, len(loss_list), 10):
            new_loss_list.append(sum(loss_list[i:i+10])/len(loss_list[i:i+10]))
        loss_list = new_loss_list
    if len(loss_list) == 0: return
    x = range(len(loss_list))
    fig = plt.gcf()
    plt.plot(x, loss_list)
    fig.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()
    return


def plot_entropy(mat, save_path, title="", xlabel=[], ylabel1=[], ylabel2=[],):
    # construct figure
    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(mat, aspect='equal', interpolation='none' , cmap='hot', vmin=0., vmax=1.0)
    plt.ylabel('Attention head', fontsize=14)
    fig.colorbar(im)
    if len(xlabel) and len(ylabel1):
        ax.set_xticks(np.arange(len(xlabel)))
        ax.set_xticklabels(xlabel)
        # Rotate and align bottom ticklabels
        plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
                 ha="right", va="center", rotation_mode="anchor")
        ax.set_yticks(np.arange(len(ylabel1)))
        ax.set_yticklabels(ylabel1)
        # Set ticks on both sides of axes on
        ax.tick_params(axis="x", bottom=True, top=True, labelleft=True, labelright=True, labeltop=True)
    ax.set_title("Layers", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()
    return


def plot_sim(mat, save_path, title="", xlabel=[], ylabel1=[], ylabel2=[], pos=False, depth=False, depth_pred=False, prune_list=[]):
    # construct figure
    fig = plt.figure()
    ax = plt.gca()
    if pos:
        ax.set_title("Layers", fontsize=14)
        im = ax.matshow(mat, aspect='equal', interpolation='none' , cmap='seismic', vmin=0., vmax=0.92)
        plt.ylabel('Position ranking', fontsize=14)
    else:
        if depth:
            ax.set_title("Layers", fontsize=14)
            xlabel = [i+1 for i in range(32)]
            if len(prune_list):
                for l in prune_list:
                    if l+1 in xlabel:
                        xlabel.remove(l+1)
            ylabel1 = [i for i in range(mat.shape[0])]
            ylabel1 = ["NA"] + ylabel1[:-1]
            im = ax.matshow(mat, aspect='equal', interpolation='none' , cmap='seismic')
            plt.ylabel('Depth', fontsize=14)
            fig.colorbar(im, location='bottom', shrink=0.5, pad=0.05)
        else:
            if depth_pred:
                dmin, omax= 1., 0.
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        if i == j:
                            dmin = min(mat[i,j], dmin)
                        else:
                            omax = max(mat[i,j], omax)
                ax.set_title("Prediction", fontsize=14)
                im = ax.matshow(mat, aspect='equal', interpolation='none' , cmap='seismic',
                    norm=colors.TwoSlopeNorm(vcenter=min(dmin, omax)+0.5*(max(dmin, omax)-min(dmin, omax))))
                plt.ylabel('GroundTruth', fontsize=14)
                fig.colorbar(im,)# shrink=0.5,)
                for (i, j), z in np.ndenumerate(mat):
                    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=11)
            else:
                ax.set_title("Layers", fontsize=14)
                im = ax.matshow(mat, aspect='equal', interpolation='none' , cmap='seismic', 
                                    norm=colors.TwoSlopeNorm(vcenter=0.077))#norm=colors.PowerNorm(gamma=0.2),)
                plt.ylabel('Size ranking', fontsize=14)
                fig.colorbar(im)
    if len(xlabel) and len(ylabel1):
        ax.set_xticks(np.arange(len(xlabel)))
        ax.set_xticklabels(xlabel)
        ax.set_yticks(np.arange(len(ylabel1)))
        ax.set_yticklabels(ylabel1)
        # Set ticks on both sides of axes on
        ax.tick_params(axis="x", bottom=True, top=True, labelleft=True, labelright=True, labeltop=True)
        if depth:
            # Rotate and align bottom ticklabels
            plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=60,
                     ha="right", va="center", rotation_mode="anchor")
            # Rotate and align top ticklabels
            plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=60,
                     ha="left", va="center",rotation_mode="anchor")
    # ax.set_title(title, pad=55)
    '''
    # plot number
    for (i, j), z in np.ndenumerate(mat):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    '''
    fig.tight_layout()
    fig.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()
    return


def plot_flow(data, save_path, title, xlabel, ylabel, cs=False, pos1=None, pos2=None):
    if len(data) == 0: return
    ax = plt.gca()
    ax.set_xticks(np.arange(len(xlabel)))
    ax.set_xticklabels(xlabel)
    ax.set_yticks(np.arange(len(ylabel)))
    ax.set_yticklabels(ylabel)
    plt.ylabel('Token', fontsize=14)
    plt.xlabel('Layer', fontsize=14)
    # Set ticks on both sides of axes on
    ax.tick_params(axis="x", bottom=False, top=False, labelleft=True, labelright=True, labeltop=False)
    # Rotate and align bottom ticklabels
    # plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45, ha="right", va="center", rotation_mode="anchor")
    if cs:
        layer_num = data.shape[0]
        token_num = data.shape[1]
        for l in range(layer_num):
            for t1 in range(token_num):
                for t2 in range(token_num):
                    if data[l, t1, t2] == 0: continue
                    # if t1 != token_num-1: continue
                    x = [l+1, l]
                    y = [token_num-1-t1, token_num-1-t2]
                    fig = plt.gcf()
                    if l == layer_num-1 and token_num-1-t1 != 0: continue
                    if pos1 is not None and pos2 is not None:
                        if t2 == pos1-1:
                            plt.plot(x, y, color='blue', alpha=data[l, t1, t2])
                        elif t2 == pos2-1:
                            plt.plot(x, y, color='red', alpha=data[l, t1, t2])
                        else:
                            plt.plot(x, y, color='grey', alpha=data[l, t1, t2])
                    else:
                        plt.plot(x, y, color='grey', alpha=data[l, t1, t2])
        # plot neuron
        nx_list, ny_list = [], []
        for i in range(layer_num+1):
            for j in range(token_num):
                if i == 12 and j != 0: continue
                nx_list.append(i)
                ny_list.append(j)
        plt.scatter(nx_list, ny_list, s=12)
    else:
        # plot flow
        for d in data:
            x = range(len(d))
            fig = plt.gcf()
            plt.plot(x, d, alpha=0.01)
            # plt.title(title)
        # plot neuron
        nx_list, ny_list = [], []
        for i in range(12):
            for j in range(16):
                nx_list.append(i)
                ny_list.append(j)
        plt.scatter(nx_list, ny_list, s=12)
    fig.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()
    return


def get_label_idx(data, label):
    label_list = []
    data_list = [d.split(" ") for d in data]
    if len(label) != len(data_list): print("Error: data length not equal to label length!")
    for i in range(len(data_list)):
        label_idx = data_list[i].index(label[i])
        label_list.append(label_idx)
    return label_list


class data_generator():
    def __init__(self, args, tokenizer):
        fix_seed(args.random_seed)
        self.max_num = args.stask_set  # 512/1800 numbers with 1 token
        self.list_len = args.stask_len # 64 for accuracy analysis, 16 for attn analysis
        self.tokenizer = tokenizer
        tmp_data_path = os.path.join(args.tmp_dir, args.stask+"_tmp_data.json")
        if os.path.exists(tmp_data_path):
            self.load_tmp_data(tmp_data_path)
        else:
            self.generate_save_data(args, tmp_data_path)

    def generate_save_data(self, args, path):
        train_data, val_data, test_data = self.generate_data(args.sdata_num)
        train_label = self.generate_label(args, train_data)
        val_label = self.generate_label(args, val_data)
        test_label = self.generate_label(args, test_data)
        # int to str
        train_sdata = [self.int2str(intl) for intl in train_data]
        val_sdata = [self.int2str(intl) for intl in val_data]
        test_sdata = [self.int2str(intl) for intl in test_data]
        train_slabel = [str(intid) for intid in train_label]
        val_slabel = [str(intid) for intid in val_label]
        test_slabel = [str(intid) for intid in test_label]
        # added to self
        self.train_data = train_sdata
        self.val_data = val_sdata
        self.test_data = test_sdata
        self.train_label = train_slabel
        self.val_label = val_slabel
        self.test_label = test_slabel
        return

    def generate_data(self, sdata_num):
        all_data = []
        rand_list = []
        count = 0
        while len(rand_list) < self.max_num:
            t = self.tokenizer(str(count)+" " + str(count), add_special_tokens=False)
            if len(t["input_ids"]) == 2:
                rand_list.append(count)
            count += 1
            if count > 1e5: break
        for _ in range(sdata_num):
            all_data.append(random.choices(rand_list, k=self.list_len))
        split_idx1 = int(len(all_data)*0.98)
        split_idx2 = int(len(all_data)*0.01) + split_idx1
        return all_data[:split_idx1], all_data[split_idx1:split_idx2], all_data[split_idx2:]

    def generate_label(self, args, data_list):
        label_list = []
        for data in data_list:
            idx = int(args.stask[-2:])
            label_list.append(sorted(data)[idx])
        return label_list

    def int2str(self, intl):
        strl = [str(n) for n in intl]
        return " ".join(strl)

    def return_data(self, data=True):
        if data:
            return self.train_data, self.val_data, self.test_data
        else:
            return self.train_label, self.val_label, self.test_label


class data_loader():
    def __init__(self, args):
        fix_seed(args.random_seed)
        train_dict, dev_dict, test_dict = self.get_data(args.data_dir)
        self.train_dict = train_dict
        self.dev_dict = dev_dict
        self.test_dict = test_dict

    def get_data(self, path):
        if path.split("/")[-1] == "proofwriter":
            path = os.path.join(path, "CWA")
            if not os.path.exists(os.path.join(path, "test.json")):
                train_dict, dev_dict, test_dict = self.preprocess_proofwriter(path)
            else:
                train_dict, dev_dict, test_dict = self.load_data(path)
        elif path.split("/")[-1] == "arc":
            if not os.path.exists(os.path.join(path, "test.json")):
                train_dict, dev_dict, test_dict = self.preprocess_arc(path)
            else:
                train_dict, dev_dict, test_dict = self.load_data(path)
        else:
            print("Warning: wrong data directory!")

        return train_dict, dev_dict, test_dict

    def preprocess_proofwriter(self, path):
        dir_list = ["depth-0", "depth-1", "depth-2", "depth-3", "depth-5"]
        train_dict = {i:[] for i in range(6)}
        dev_dict = {i:[] for i in range(6)}
        test_dict = {i:[] for i in range(6)}
        for d in dir_list:
            tmp_path_train = os.path.join(path, d, "meta-train.jsonl")
            tmp_path_dev = os.path.join(path, d, "meta-dev.jsonl")
            tmp_path_test = os.path.join(path, d, "meta-test.jsonl")
            # process data
            with open(tmp_path_train, "r") as f:
                for line in f:
                    tmp_data = json.loads(line)
                    # clean data
                    c = tmp_data["theory"]
                    t, r = {}, {}
                    for k, v in tmp_data["triples"].items(): t[k] = v["text"]
                    for k, v in tmp_data["rules"].items(): r[k] = v["text"]
                    q_count = 0
                    for qid, qdict in tmp_data["questions"].items():
                        if qdict["QLen"] == "": continue  # ignore q without proof
                        q = qdict["question"]
                        a = qdict["answer"]
                        p = qdict["proofs"]
                        if "NAF" in p: continue  # ignore proof with negation-as-failure
                        q_count += 1
                        # add data
                        new_data = {}
                        new_data["context"] = c
                        new_data["triples"] = t
                        new_data["rules"] = r
                        new_data["question"] = q
                        new_data["answer"] = a
                        new_data["proof"] = p
                        q_depth = qdict["QDep"]
                        train_dict[q_depth].append(new_data)
            with open(tmp_path_dev, "r") as f:
                for line in f:
                    tmp_data = json.loads(line)
                    # clean data
                    c = tmp_data["theory"]
                    t, r = {}, {}
                    for k, v in tmp_data["triples"].items(): t[k] = v["text"]
                    for k, v in tmp_data["rules"].items(): r[k] = v["text"]
                    q_count = 0
                    for qid, qdict in tmp_data["questions"].items():
                        if qdict["QLen"] == "": continue  # ignore q without proof
                        q = qdict["question"]
                        a = qdict["answer"]
                        p = qdict["proofs"]
                        if "NAF" in p: continue  # ignore proof with negation-as-failure
                        q_count += 1
                        # add data
                        new_data = {}
                        new_data["context"] = c
                        new_data["triples"] = t
                        new_data["rules"] = r
                        new_data["question"] = q
                        new_data["answer"] = a
                        new_data["proof"] = p
                        q_depth = qdict["QDep"]
                        dev_dict[q_depth].append(new_data)
            with open(tmp_path_test, "r") as f:
                for line in f:
                    tmp_data = json.loads(line)
                    # clean data
                    c = tmp_data["theory"]
                    t, r = {}, {}
                    for k, v in tmp_data["triples"].items(): t[k] = v["text"]
                    for k, v in tmp_data["rules"].items(): r[k] = v["text"]
                    q_count = 0
                    for qid, qdict in tmp_data["questions"].items():
                        if qdict["QLen"] == "": continue  # ignore q without proof
                        q = qdict["question"]
                        a = qdict["answer"]
                        p = qdict["proofs"]
                        if "NAF" in p: continue  # ignore proof with negation-as-failure
                        q_count += 1
                        # add data
                        new_data = {}
                        new_data["context"] = c
                        new_data["triples"] = t
                        new_data["rules"] = r
                        new_data["question"] = q
                        new_data["answer"] = a
                        new_data["proof"] = p
                        q_depth = qdict["QDep"]
                        test_dict[q_depth].append(new_data)
        # save data
        with open(os.path.join(path, "train.json"), "w") as f:
            f.write(json.dumps(train_dict))
        with open(os.path.join(path, "dev.json"), "w") as f:
            f.write(json.dumps(dev_dict))
        with open(os.path.join(path, "test.json"), "w") as f:
            f.write(json.dumps(test_dict))
        print("Training data number: ", {k:len(v) for k, v in train_dict.items()})
        print("Dev data number: ", {k:len(v) for k, v in dev_dict.items()})
        print("Test data number: ", {k:len(v) for k, v in test_dict.items()})
        return train_dict, dev_dict, test_dict

    def preprocess_arc(self, path):
        dir_list = ["depth-0", "depth-1", "depth-2", "depth-3", "depth-5"]
        train_dict = {}
        dev_dict = {}
        test_dict = {}
        tmp_path_train = os.path.join(path, "reasoning_annotated_train.jsonl")
        tmp_path_dev = os.path.join(path, "reasoning_annotated_dev.jsonl")
        tmp_path_test = os.path.join(path, "reasoning_annotated_test.jsonl")
        # process data
        with open(tmp_path_train, "r") as f:
            for line in f:
                tmp_data = json.loads(line)
                # add data
                new_data = {}
                new_data["context"] = tmp_data["context"]
                all_triples = {str(k):v for k, v in tmp_data["textual_logical_units"].items()}
                new_all_triples = {}
                anwser_key = "1"
                for k, v in all_triples.items():
                    if "The answer is" not in v:
                        new_all_triples[k] = v
                    else:
                        anwser_key = k
                new_data["triples"] = new_all_triples
                new_data["rules"] = {}
                new_data["options"] = tmp_data["options"]
                new_data["question"] = tmp_data["question"] + " ".join(tmp_data["options"])
                new_data["answer"] = tmp_data["answer"][0]
                all_edges = tmp_data["reasoning_graph_edges"]
                tmp_g = nx.DiGraph()
                input_nodes = set()
                for e in all_edges:
                    for a in e["antecedents"]:
                        if int(a) < int(anwser_key) and int(e["consequent"]) < int(anwser_key):
                            tmp_g.add_edge(str(a), str(e["consequent"]))
                            input_nodes.add(str(a))
                            input_nodes.add(str(e["consequent"]))
                proof = ""
                g_nodes = dag_longest_path(tmp_g)
                g_input = g_nodes[0]
                for n in input_nodes:
                    if n not in g_nodes:
                        g_input = n + " " + g_input
                g_input = "(" + g_input + ")"
                g_nodes[0] = g_input
                proof = " -> ".join(g_nodes)
                new_data["proof"] = proof
                q_depth = len(g_nodes) - 1
                if q_depth not in train_dict: train_dict[q_depth] = []
                train_dict[q_depth].append(new_data)
        with open(tmp_path_dev, "r") as f:
            for line in f:
                tmp_data = json.loads(line)
                # add data
                new_data = {}
                new_data["context"] = tmp_data["context"]
                all_triples = {str(k):v for k, v in tmp_data["textual_logical_units"].items()}
                new_all_triples = {}
                anwser_key = "1"
                for k, v in all_triples.items():
                    if "The answer is" not in v:
                        new_all_triples[k] = v
                    else:
                        anwser_key = k
                new_data["triples"] = new_all_triples
                new_data["rules"] = {}
                new_data["options"] = tmp_data["options"]
                new_data["question"] = tmp_data["question"] + " ".join(tmp_data["options"])
                new_data["answer"] = tmp_data["answer"][0]
                all_edges = tmp_data["reasoning_graph_edges"]
                tmp_g = nx.DiGraph()
                input_nodes = set()
                for e in all_edges:
                    for a in e["antecedents"]:
                        if int(a) < int(anwser_key) and int(e["consequent"]) < int(anwser_key):
                            tmp_g.add_edge(str(a), str(e["consequent"]))
                            input_nodes.add(str(a))
                            input_nodes.add(str(e["consequent"]))
                proof = ""
                g_nodes = dag_longest_path(tmp_g)
                g_input = g_nodes[0]
                for n in input_nodes:
                    if n not in g_nodes:
                        g_input = n + " " + g_input
                g_input = "(" + g_input + ")"
                g_nodes[0] = g_input
                proof = " -> ".join(g_nodes)
                new_data["proof"] = proof
                q_depth = len(g_nodes) - 1
                if q_depth not in dev_dict: dev_dict[q_depth] = []
                dev_dict[q_depth].append(new_data)
        with open(tmp_path_test, "r") as f:
            for line in f:
                tmp_data = json.loads(line)
                # add data
                new_data = {}
                new_data["context"] = tmp_data["context"]
                all_triples = {str(k):v for k, v in tmp_data["textual_logical_units"].items()}
                new_all_triples = {}
                anwser_key = "1"
                for k, v in all_triples.items():
                    if "The answer is" not in v:
                        new_all_triples[k] = v
                    else:
                        anwser_key = k
                new_data["triples"] = new_all_triples
                new_data["rules"] = {}
                new_data["options"] = tmp_data["options"]
                new_data["question"] = tmp_data["question"] + " ".join(tmp_data["options"])
                new_data["answer"] = tmp_data["answer"][0]
                all_edges = tmp_data["reasoning_graph_edges"]
                tmp_g = nx.DiGraph()
                input_nodes = set()
                for e in all_edges:
                    for a in e["antecedents"]:
                        if int(a) < int(anwser_key) and int(e["consequent"]) < int(anwser_key):
                            tmp_g.add_edge(str(a), str(e["consequent"]))
                            input_nodes.add(str(a))
                            input_nodes.add(str(e["consequent"]))
                proof = ""
                g_nodes = dag_longest_path(tmp_g)
                g_input = g_nodes[0]
                for n in input_nodes:
                    if n not in g_nodes:
                        g_input = n + " " + g_input
                g_input = "(" + g_input + ")"
                g_nodes[0] = g_input
                proof = " -> ".join(g_nodes)
                new_data["proof"] = proof
                q_depth = len(g_nodes) - 1
                if q_depth not in test_dict: test_dict[q_depth] = []
                test_dict[q_depth].append(new_data)
        # save data
        with open(os.path.join(path, "train.json"), "w") as f:
            f.write(json.dumps(train_dict))
        with open(os.path.join(path, "dev.json"), "w") as f:
            f.write(json.dumps(dev_dict))
        with open(os.path.join(path, "test.json"), "w") as f:
            f.write(json.dumps(test_dict))
        print("Training data number: ", {k:len(v) for k, v in train_dict.items()})
        print("Dev data number: ", {k:len(v) for k, v in dev_dict.items()})
        print("Test data number: ", {k:len(v) for k, v in test_dict.items()})
        return train_dict, dev_dict, test_dict

    def load_data(self, path):
        with open(os.path.join(path, "train.json"), "r") as f:
            train_dict = json.loads(f.read())
        with open(os.path.join(path, "dev.json"), "r") as f:
            dev_dict = json.loads(f.read())
        with open(os.path.join(path, "test.json"), "r") as f:
            test_dict = json.loads(f.read())
        print("Training data number: ", {k:len(v) for k, v in train_dict.items()})
        print("Dev data number: ", {k:len(v) for k, v in dev_dict.items()})
        print("Test data number: ", {k:len(v) for k, v in test_dict.items()})
        return train_dict, dev_dict, test_dict

    def return_data(self):
        return self.train_dict, self.dev_dict, self.test_dict

    def return_shuffled_data(self):
        train_data, dev_data, test_data = [], [], []
        for k, v in self.train_dict.items(): train_data = train_data + v
        for k, v in self.dev_dict.items(): dev_data = dev_data + v
        for k, v in self.test_dict.items(): test_data = test_data + v
        random.shuffle(train_data)
        random.shuffle(dev_data)
        random.shuffle(test_data)
        return train_data, dev_data, test_data