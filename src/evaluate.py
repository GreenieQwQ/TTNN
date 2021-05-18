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

def syntactic_acc(pred: str, gd: str):
    refined_gd = gd.replace("\"", "").replace(",", ";")
    result = (pred == refined_gd)
    # if not result:
    #     print(f"pred: {pred}")
    #     print(f"refined_gd: {refined_gd}")
    #     print()
    return result


def semantic_acc(pred: str, df):
    # refined_pred = pred.replace(";", ",")
    vocab = [i for i in "abcdefghij"]
    # print(f"Vocab is: {vocab}.")
    return check(df['ltl'], pred, vocab)
    # return runAutom(df['APs'], df['States'],
    #                 df['Transform'], df['Accept'], df['Start'], refined_pred)


# 对含有#的进行处理 取#之前的部分
def processPred(pred: str):
    pred = pred.strip()
    splitToken = "#"
    processed = pred.split(splitToken)
    return processed[0]


def evaluate(pred_path, gd_path, output_path):
    print(f"Evaluating at {gd_path}")
    print(f"Prediction at {pred_path}")
    if not os.path.isfile(pred_path):
        print(f"Err: {pred_path} does not exist.")
        return

    if not os.path.isfile(gd_path):
        print(f"Err {gd_path} does not exist.")
        return

    total_count = 0
    syntactic_count = 0
    semantic_count = 0

    gd_dataframe = pd.read_json(gd_path)
    result = {}
    with open(pred_path, 'r', encoding='utf-8') as predFile:
        for i, pred in tqdm(enumerate(predFile), desc="Evaluating"):
            pred = processPred(pred)  # important
            total_count += 1
            gd = gd_dataframe.loc[i]
            result[i] = {'ltl_pre': gd['ltl_pre']}
            if syntactic_acc(pred, gd['trace']):
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

    output = f"Total: {total_count}"
    print_and_save(output)
    output = f"Syntactic accuracy: {syntactic_count / total_count}"
    print_and_save(output)
    output = f"Semantic accuracy: {semantic_count / total_count}"
    print_and_save(output)
    output = f"Total accuracy: {(syntactic_count + semantic_count) / total_count}"
    print_and_save(output)
    output = f"[Info] Done at {pred_path}."
    print_and_save(output)
    with open(output_path, 'w', encoding='utf-8') as o:
        o.writelines([l + '\n' for l in result_strs])


if __name__ == "__main__":
    range_names = ["5t20", "20t35", "35t50", "50t65", "65t80"]
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
