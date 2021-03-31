import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--pred', required=True,
                    help='Path of the prediction.')
parser.add_argument('--src', required=True,
                    help='Path of the source.')
parser.add_argument('--gd', required=True,
                    help='Path of the ground truth.')

args = parser.parse_args()


def syntactic_acc(pred: str, gd: str):
    return pred == gd


def semantic_acc(pred: str, ltl: str):
    return True
    return is_satisfying_trace(pred, ltl)


def is_satisfying_trace(trace: str, ltl: str):
    raise NotImplemented("Need to implement model checking technique.")


if __name__ == "__main__":
    total_count = 0
    syntactic_count = 0
    semantic_count = 0
    with open(args.pred, 'r', encoding='utf-8') as predFile:
        with open(args.src, 'r', encoding='utf-8') as srcFile:
            with open(args.gd, 'r', encoding='utf-8') as gdFile:
                for pred, src, gd in tqdm(zip(predFile, srcFile, gdFile), desc="Evaluating"):
                    total_count += 1
                    if syntactic_acc(pred, gd):
                        syntactic_count += 1
                    elif semantic_acc(pred, src):
                        semantic_count += 1
    # endwith
    print(f"Total: {total_count}")
    print(f"Syntactic accuracy: {syntactic_count/total_count}")
    print(f"Semantic accuracy: {semantic_count / total_count}")
    print(f"Total accuracy: {(syntactic_count+ semantic_count) / total_count}")
