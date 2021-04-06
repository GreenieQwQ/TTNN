import argparse
import pandas as pd
from tqdm import tqdm
from runautom import runAutom

parser = argparse.ArgumentParser()
parser.add_argument('--pred', required=True,
                    help='Path of the prediction.')
# parser.add_argument('--src', required=True,
#                     help='Path of the source.')
parser.add_argument('--gd', required=True,
                    help='Path of the ground truth.')

args = parser.parse_args()


def syntactic_acc(pred: str, gd: str):
    return pred == gd


def semantic_acc(pred: str, df: pd.dataframe):
    return runAutom(df['APs'], df['States'],
                    df['Transform'], df['Accept'], df['Start'], pred)


if __name__ == "__main__":
    total_count = 0
    syntactic_count = 0
    semantic_count = 0
    
    gd_dataframe = pd.read_json(args.gd)

    with open(args.pred, 'r', encoding='utf-8') as predFile:
        for i, pred in tqdm(enumerate(predFile), desc="Evaluating"):
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
