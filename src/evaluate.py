import argparse
import os

import pandas as pd
from tqdm import tqdm
from ltl_model_check import check
from multiprocessing import Process
parser = argparse.ArgumentParser()
parser.add_argument("--dn", type=str, required=True, help="data name")
parser.add_argument("--rn", type=str, required=True, help="range name")
parser.add_argument("--tdn", type=str, default="", help="target data name")
parser.add_argument("--epoch", type=str, default="", help="model epoch")
parser.add_argument('--leak', action='store_true')
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
    refined_gd = gd.replace("\"", "").replace(",", ";")
    result = (pred == refined_gd)
    return result


def semantic_acc(pred: str, df):
    # refined_pred = pred.replace(";", ",")
    vocab = [i for i in "abcdefghij"]
    # print(f"Vocab is: {vocab}.")
    ltl = df['ltl']
    return check(ltl, pred, vocab)


# 对含有#的进行处理 取#之前的部分
def processPred(pred: str):
    pred = pred.strip()
    if pred.count(",") == 0:
        splitToken = "#"
        processed = pred.split(splitToken)
        return processed[0]
    else:
        splitToken = ","
        processed = pred.split(splitToken)
        return processed[1]

is_proof_process = True
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
    semantic_count = 0

    gd_dataframe = pd.read_json(gd_path)
    if pred_path.count("balance") > 0:
        path = gd_path.replace("spot_balance", "spot")
        ltlDf = pd.read_json(path)
        print(f"Using ltl df form {path}.")
    if pred_path.count("one") > 0:
        path = gd_path.replace("spot_one", "spot")
        ltlDf = pd.read_json(path)
        print(f"Using ltl df form {path}.")

    # 预测用的src
    dir_name = gd_path.replace("/origin-test.json", "")
    input_path = os.path.join(dir_name, "src-test.txt")

    result = {}
    with open(pred_path, 'r', encoding='utf-8') as predFile:
        with open(input_path, 'r', encoding='utf-8') as inFile:
            for i, data in tqdm(enumerate(zip(predFile, inFile)), desc="Evaluating"):
                pred, src = data
                # finding leaked data
                def processSrc(seq):
                    seq = seq.strip()
                    def is_proof(name):
                        return name.count("balance") > 0 or name.count("one") > 0
                    if is_proof(data_name) and not is_proof(tgt_data_name):
                        seq += ",$,1"
                        global is_proof_process
                        if is_proof_process:
                            print("processing")
                            is_proof_process = False
                    return seq
                # enddef
                src = processSrc(src)
                pred = processPred(pred)  # important
                if not args.leak and src in train_set:
                    leak_count += 1
                    continue

                total_count += 1
                gd = gd_dataframe.loc[i]
                try:
                    result[i] = {'ltl_pre': gd['ltl_pre']}
                except KeyError:
                    result[i] = {}
                try:
                    trace = gd['trace']
                except KeyError:
                    trace = gd['tgt'].split(',')[1]
                    # 使用另外一个gd获取ltl
                    gd = ltlDf.loc[i]
                if syntactic_acc(pred, trace):
                    syntactic_count += 1
                    result[i]['state'] = "syntactic"
                elif semantic_acc(pred, gd):
                    semantic_count += 1
                    result[i]['state'] = "semantic"
                else:
                    result[i]['state'] = "False"
                # endif

                if total_count == 100000:
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
    output = f"Syntactic accuracy: {syntactic_count / total_count}"
    print_and_save(output)
    output = f"Semantic accuracy: {semantic_count / total_count}"
    print_and_save(output)
    output = f"Total accuracy: {(syntactic_count + semantic_count) / total_count}"
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
