import argparse
import os

import pandas as pd
from tqdm import tqdm
from runautom import runAutom
from ltl_model_check import check

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
args = parser.parse_args()


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
    print(f"Vocab is: {vocab}.")
    return check(df['ltl'], pred, vocab)
    # return runAutom(df['APs'], df['States'],
    #                 df['Transform'], df['Accept'], df['Start'], refined_pred)


def evaluate(pred_path, gd_path):
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
            pred = pred.strip()  # important
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
    # endwith
    # import json
    # f = open(path, 'w')
    # json.dump(result, f)
    # resultPath = "../data/result_statistics"
    # if not os.path.isdir(resultPath):
    #     os.makedirs(resultPath)
    # path = os.path.join(resultPath, f"{prefix}src-{postfix}_result.json")
    # df = pd.DataFrame(result)
    # df.to_json(path)
    # print(f"Writing result to {path}.")

    print(f"Total: {total_count}")
    print(f"Syntactic accuracy: {syntactic_count / total_count}")
    print(f"Semantic accuracy: {semantic_count / total_count}")
    print(f"Total accuracy: {(syntactic_count + semantic_count) / total_count}")
    print(f"[Info] Done at {pred_path}.")


if __name__ == "__main__":
    # isChallenge = args.challenge
    # mode = args.mode
    # print(f"Postfix: {args.postfix}\tChallenge: {isChallenge}\tMode: {mode}")

    # prefixes = ["5t20", "20t35", "35t50", "50t65", "65t80", "80t105"]
    # gd_dir = "../data/test"
    # gd_name = "ltl{prefix}-origin-test.json" if not isChallenge else "{prefix}ltl-origin-{mode}.json"
    # pred_dir = "../data/prediction"
    # pred_name = "{prefix}src-{postfix}.txt" if not isChallenge else "{prefix}ltl-{mode}-{postfix}ltl.txt"
    # postfix = args.postfix
    # for prefix in prefixes:
    #     gd_path = os.path.join(gd_dir, gd_name.format(prefix=prefix, mode=mode))
    #     pred_path = os.path.join(pred_dir, pred_name.format(prefix=prefix, postfix=postfix, mode=mode))
    #     print(f"Evaluating at {gd_path}")
    #     print(f"Prediction at {pred_path}")
    #
    #     total_count = 0
    #     syntactic_count = 0
    #     semantic_count = 0
    #
    #     gd_dataframe = pd.read_json(gd_path)
    #
    #     resultPath = "../data/result_statistics"
    #     if not os.path.isdir(resultPath):
    #         os.makedirs(resultPath)
    #     path = os.path.join(resultPath, f"{prefix}src-{postfix}_result.json")
    #     result = {}
    #     with open(pred_path, 'r', encoding='utf-8') as predFile:
    #         for i, pred in tqdm(enumerate(predFile), desc="Evaluating"):
    #             pred = pred.strip()     # important
    #             total_count += 1
    #             gd = gd_dataframe.loc[i]
    #             result[i] = {'ltl_pre': gd['ltl_pre']}
    #             if syntactic_acc(pred, gd['trace']):
    #                 syntactic_count += 1
    #                 result[i]['state'] = "syntactic"
    #             elif semantic_acc(pred, gd):
    #                 semantic_count += 1
    #                 result[i]['state'] = "semantic"
    #             else:
    #                 result[i]['state'] = "False"
    #     # endwith
    #     # import json
    #     # f = open(path, 'w')
    #     # json.dump(result, f)
    #
    #     # df = pd.DataFrame(result)
    #     # df.to_json(path)
    #     # print(f"Writing result to {path}.")
    #
    #     print(f"Total: {total_count}")
    #     print(f"Syntactic accuracy: {syntactic_count/total_count}")
    #     print(f"Semantic accuracy: {semantic_count / total_count}")
    #     print(f"Total accuracy: {(syntactic_count+ semantic_count) / total_count}")

    data_name = args.dn
    range_name = args.rn
    range_names = ["5t20", "20t35", "35t50", "50t65", "65t80"]
    for rn in range_names:
        pred_dir = f"../data/prediction-{data_name}-{range_name}"
        pred_name = f"prediction-{data_name}-{rn}"  # 暂定统一分布 即dn
        pred_path = os.path.join(pred_dir, pred_name)
        raw_path = f"../data/{data_name}-{rn}-raw/origin-test.json"
        evaluate(pred_path, raw_path)
    # endfor
    print("[Info] All Done.")
