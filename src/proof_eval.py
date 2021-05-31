import argparse
import os

import pandas as pd
from tqdm import tqdm
from runautom import runAutom
from ltl_model_check import check
from multiprocessing import Process
parser = argparse.ArgumentParser()
# parser.add_argument('--pred', required=True,
#                     help='Path of the prediction.')
# parser.add_argument('--src', required=True,
#                     help='Path of the source.')
# parser.add_argument('--gd', required=True,
#                     help='Path of the ground truth.')
# parser.add_argument('--challenge', action="store_true",
#                     help='Activate challenge mode.')
# parser.add_argument('--postfix', required=True,
#                     help='Model name.')
# parser.add_argument('--mode', type=str, required=True)
parser.add_argument("--dn", type=str, required=True, help="data name")
parser.add_argument("--rn", type=str, required=True, help="range name")
parser.add_argument("--tdn", type=str, default="", help="target data name")
parser.add_argument("--epoch", type=str, default="", help="model epoch")
args = parser.parse_args()

data_name = args.dn
range_name = args.rn
tgt_data_name = args.dn if args.tdn == "" else args.tdn
epoch_postfix = ("_" + args.epoch) if args.epoch != "" else ""
output_dir = f"../eval/{args.dn}-{args.rn}{epoch_postfix}"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
print(f"Result at {output_dir}.")

# 查找数据泄露的index
print("Building train set...")
# gd_path : f"../data/{tgt_data_name}-{rn}-raw/origin-test.json"
# 处理src文件
src_dir = f"../data/{args.dn}-{args.rn}"
f_list = os.listdir(src_dir)
src_path = ""
for f in f_list:
    if f.endswith("train.json"):
        src_path = f
    # end
# endfor
if src_path == "":
    print(f"Invalid src dir {src_dir}.")
    exit()
# endif

src_path = os.path.join(src_dir, src_path)
src_df = pd.read_json(src_path)
# 判断种类
dummy = src_df.loc[0]
try:
    source = dummy['ltl_pre'].strip().replace("->", "I")
    train_set = set(src_df['ltl_pre'])
    print("Training set using ltl_pre.")
except KeyError:
    source = dummy['src'].strip().replace("->", "I")
    train_set = set(src_df['src'])
    print("Training set using src.")

def syntactic_acc(pred: str, gd: str):
    result = (pred == gd)
    return result

def evaluate(pred_path, gd_path, output_path):
    print(f"Evaluating at {gd_path}")
    print(f"Prediction at {pred_path}")
    if not os.path.isfile(pred_path):
        print(f"Err: {pred_path} does not exist.")
        return

    if not os.path.isfile(gd_path):
        print(f"Err {gd_path} does not exist.")
        return

    leak_count = 0
    total_count = 0
    syntactic_count = 0
    gd_dataframe = pd.read_json(gd_path)

    # 预测用的src
    dir_name = gd_path.replace("/origin-test.json", "")
    input_path = os.path.join(dir_name, "src-test.txt")

    with open(pred_path, 'r', encoding='utf-8') as predFile:
        with open(input_path, 'r', encoding='utf-8') as inFile:
            for i, data in tqdm(enumerate(zip(predFile, inFile)), desc="Evaluating"):
                pred, src = data
                # finding leaked data
                def process(seq):
                    seq = seq.strip()
                    return seq
                # enddef
                src = process(src)
                pred = process(pred)  # important
                if src in train_set:
                    leak_count += 1
                    continue

                total_count += 1
                gd = gd_dataframe.loc[i]
                try:
                    trace = gd['trace']
                except KeyError:
                    trace = gd['tgt']
                if syntactic_acc(pred, trace):
                    syntactic_count += 1
                # endif
                if i >= int(1e5):
                    break
    # endwith

    result_strs = []
    def print_and_save(out):
        print(out)
        result_strs.append(out)

    output = f"Leak count: {leak_count}"
    print_and_save(output)
    output = f"Total: {total_count}"
    print_and_save(output)
    output = f"Total accuracy: {syntactic_count / total_count}"
    print_and_save(output)
    output = f"[Info] Done at {pred_path}. Output to {output_path}."
    print_and_save(output)
    with open(output_path, 'w', encoding='utf-8') as o:
        o.writelines([l + '\n' for l in result_strs])


if __name__ == "__main__":
    range_names = ["5t20", "20t35", "35t50"]
    processes = []
    for rn in range_names:
        output_name = f"{tgt_data_name}-{rn}"
        output_path = os.path.join(output_dir, output_name)

        pred_dir = f"../data/prediction-{data_name}-{range_name}{epoch_postfix}"
        pred_name = f"prediction-{tgt_data_name}-{rn}.txt"
        pred_path = os.path.join(pred_dir, pred_name)

        raw_path = f"../data/{tgt_data_name}-{rn}-raw/origin-test.json"

        p = Process(target=evaluate, args=(pred_path, raw_path, output_path))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    # endfor
    print("[Info] All Done.")
