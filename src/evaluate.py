import argparse
import os

import pandas as pd
from tqdm import tqdm
from runautom import runAutom

parser = argparse.ArgumentParser()
# parser.add_argument('--pred', required=True,
#                     help='Path of the prediction.')
# parser.add_argument('--src', required=True,
#                     help='Path of the source.')
# parser.add_argument('--gd', required=True,
#                     help='Path of the ground truth.')
parser.add_argument('--postfix', required=True,
                    help='model name')
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
    refined_pred = pred.replace(";", ",")
    return runAutom(df['APs'], df['States'],
                    df['Transform'], df['Accept'], df['Start'], refined_pred)


if __name__ == "__main__":
    prefixes = ["5t20", "20t35", "35t50", "50t65", "65t80"]
    gd_dir = "../data/test"
    gd_name = "{prefix}ltl-origin-test.json"
    pred_dir = "../data/prediction"
    pred_name = "{prefix}ltl-{postfix}.txt"
    postfix = args.postfix
    for prefix in prefixes:
        gd_path = os.path.join(gd_dir, gd_name.format(prefix=prefix))
        pred_path = os.path.join(pred_dir, pred_name.format(prefix=prefix, postfix=postfix))
        print(f"Evaluating at {gd_path}")

        total_count = 0
        syntactic_count = 0
        semantic_count = 0

        gd_dataframe = pd.read_json(gd_path)

        with open(pred_path, 'r', encoding='utf-8') as predFile:
            for i, pred in tqdm(enumerate(predFile), desc="Evaluating"):
                pred = pred.strip()     # important
                total_count += 1
                gd = gd_dataframe.loc[i]
                if syntactic_acc(pred, gd['trace']):
                    syntactic_count += 1
                elif semantic_acc(pred, gd):
                    semantic_count += 1
        # endwith
        print(f"Total: {total_count}")
        print(f"Syntactic accuracy: {syntactic_count/total_count}")
        print(f"Semantic accuracy: {semantic_count / total_count}")
        print(f"Total accuracy: {(syntactic_count+ semantic_count) / total_count}")
    print("[Info] All Done.")
