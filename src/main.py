import os
import argparse

from utils import fix_seed
from gpt2 import finetune_gpt2

from stask.attn import attn_analysis
from stask.probe import probe_analysis, probe_layer
from stask.prune import pruning_analysis, acc_calculation

from rtask.proofwriter_probe import proofwriter_probe_analysis, proofwriter_corrupt_analysis
from rtask.arc_probe import arc_probe_analysis


def main_func(args):
    fix_seed(args.random_seed)
    if args.analysis_task == "finetune_ksmallest":
        finetune_gpt2(args)
    elif args.analysis_task == "attn_ksmallest":
        attn_analysis(args)
        attn_analysis(args, pos=True)
    elif args.analysis_task == "probing_ksmallest":
        probe_analysis(args)
        probe_layer(args)
    elif args.analysis_task == "causal_ksmallest":
        pruning_analysis(args)
    elif args.analysis_task == "probing_proofwriter":
        proofwriter_probe_analysis(args)
    elif args.analysis_task == "corrupt_proofwriter":
        proofwriter_corrupt_analysis(args)
    elif args.analysis_task == "probing_arc":
        arc_probe_analysis(args)
    else:
        raise Exception("Error analysis task name")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MechanisticProbe")
    parser.add_argument("--analysis_task", type=str, default="finetune_ksmallest",
                        help="Analysis task: [finetune_ksmallest, attn_ksmallest, probing_ksmallest, causal_ksmallest,\
                                 probing_proofwriter, corrupt_proofwriter, probing_arc]")
    # basics
    parser.add_argument("--tmp_dir", type=str, default="./tmp/",
                        help="the cache directory")
    parser.add_argument("--task_format", type=str, default="stask",
                        help="the task for evaluation: [stask, rtask]")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed for reproducibility.")
    parser.add_argument("--device", type=int, default=0,
                        help="which GPU to use: set -1 to use CPU/prompt")
    parser.add_argument("--model_path", type=str, default="gpt2",
                        help="the base model path, can be ./ by default ([llama-7b-hf, gpt2])")
    parser.add_argument("--epoch_num", type=int, default=2,  # 2 for GPT-2
                        help="the number of training epochs for fine-tuning LLM")

    # synthetic experiment (GPT-2)
    parser.add_argument("--tuning_param", type=str, default="all",
                        help="the task of synthetic data: [all, attn, mlp, ln].")
    parser.add_argument("--sdata_num", type=int, default=int(1e6),
                        help="the number of synthetic data.")
    parser.add_argument("--stask_len", type=int, default=16,
                        help="the len of input list.")
    parser.add_argument("--stask_set", type=int, default=512,
                        help="the sampling set size of input list.")
    parser.add_argument("--stask", type=str, default="min01",
                        help="the task of synthetic data: [min00,..., ].")

    # real experiment (LLAMA)
    parser.add_argument("--dataset", type=str, default="proofwriter",
                        help="the corresponding dataset for evaluation")
    parser.add_argument("--data_dir", type=str, default="./data/proofwriter",
                        help="the data directory, can be ./ by default")
    parser.add_argument("--rtask", type=str, default="5",
                        help="the depth of reasoning tree: [0, 1, 2, 3, 4, 5]")
    parser.add_argument("--icl_num", type=int, default=4,
                        help="the number of examples for in-context learning prompt.")

    args = parser.parse_args()
    print(args)

    fix_seed(args.random_seed)

    if not os.path.exists(args.tmp_dir):
        os.mkdir(args.tmp_dir)

    main_func(args)
