import os
import pandas as pd
from ltlf2dfa.parser.ltlf import LTLfParser, LTLfAnd, LTLfUntil, LTLfNot, LTLfAlways, LTLfAtomic, LTLfNext
from sklearn.utils import shuffle
from tqdm import trange, tqdm
import re

seed = 114514

# 输入：存储数据的json、分裂的ratio
# 输出：训练集trainDataSet、验证集validateDataSet、测试集testDataSet
def processData(df, train_ratio=0.8, validation_ratio=0.1):
    total_count = len(df)
    trainSet_bound = int(total_count * train_ratio)
    validateSet_bound = int(total_count * (validation_ratio + train_ratio))
    trainSet = df.iloc[:trainSet_bound, :]
    validateSet = df.iloc[trainSet_bound:validateSet_bound, :]
    testSet = df.iloc[validateSet_bound:, :]
    return trainSet, validateSet, testSet

# 特定要求的process
def process_spec(df, testNum=100000, tr_ratio=0.9):
    total_count = len(df)
    left_count = total_count - testNum
    val_bound = int(left_count * tr_ratio)
    test_bound = total_count - testNum
    trainSet = df.iloc[:val_bound, :]
    validateSet = df.iloc[val_bound:test_bound, :]
    testSet = df.iloc[test_bound:, :]
    return trainSet, validateSet, testSet

# usage: 2prefix
def ltl2prefix(ltl: str):
    parser = LTLfParser()
    formula = parser(ltl.replace('1', 'true').replace('0', 'false'))
    return preorder(formula).replace('true', '1').replace('false', '0')

def preorder(f):
    if isinstance(f, LTLfAtomic):
        return f.s
    if isinstance(f, LTLfAnd) or isinstance(f, LTLfUntil):
        result = f.operator_symbol
        result += preorder(f.formulas[0])
        result += preorder(f.formulas[1])
        return result
    if isinstance(f, LTLfNot) or isinstance(f, LTLfNext) or isinstance(f, LTLfAlways):
        result = f.operator_symbol
        result += preorder(f.f)
        return result

# usage: write the traces and ltls in df to path
def writeData(d, src_path, frac=1):
    # reset index
    df = d.reset_index(drop=True)
    tgt_path = src_path.replace("src", "tgt")
    # containing the automata
    origin_path = src_path.replace("src", "origin").replace(".txt", ".json")
    origin_index = []
    with open(src_path, 'w', encoding='utf-8') as src:
        with open(tgt_path, 'w', encoding='utf-8') as tgt:
            for i in trange(len(df) * frac, desc="writing"):
                data = df.loc[i]
                if isinstance(data['ltl_pre'], str):
                    try:
                        target = data['tgt'].replace("\"", "").replace(",", ";")
                    except IndexError:
                        target = data['trace'].replace("\"", "").replace(",", ";")
                    # end
                    if len(target) <= 1024:
                        tgt.write(target + '\n')
                        src.write(data['ltl_pre'].strip().replace("->", "I") + '\n')
                        # src.write(ltl2prefix(data['ltl'].strip()) + '\n')
                        origin_index.append(i)
                    # endif
    # endwith
    # write origin json
    df.loc[origin_index].reset_index(drop=True).to_json(origin_path)
    print(f"Writing to {src_path}...")
    # endwith

# aggregate data
def fetch_all(file_dir="../data/random_data", outputName="shuffled.json"):
    file_list = os.listdir(file_dir)
    outputPath = os.path.join(file_dir, outputName)
    if not os.path.isfile(outputPath):
        df = None
        for f in tqdm(file_list, desc="Aggregating data"):
            if f.endswith(".json"):
                f_path = os.path.join(file_dir, f)
                if df is None:
                    df = filter_data(pd.read_json(f_path))
                else:
                    # print(f_path)
                    d_ = filter_data(pd.read_json(f_path))
                    df = pd.concat([df, d_])
            # endif
        # endfor
        # shuffle data
        df = shuffle(df, random_state=seed).reset_index(drop=True)
        # save data for test
        df.to_json(outputPath)
    else:
        df = pd.read_json(outputPath)
    return df

def fetch_mix():
    mix1_path = "../data/pattern_data/shuffled.json"
    mix2_path = "../data/random_data/shuffled.json"
    df1 = pd.read_json(mix1_path)
    df2 = pd.read_json(mix2_path)

    df = pd.concat([df1, df2])
    return df.sample(frac=0.5, random_state=seed).reset_index(drop=True)


# aggregate data
def fetch_spec(prefix="ltl5t20", file_dir="../data/random_ltltrace2", outputName="shuffled.json"):
    file_list = os.listdir(file_dir)
    file_list = [f for f in file_list if f.startswith(prefix)]
    outputName = ("_" + prefix + ".").join(outputName.split("."))
    outputPath = os.path.join(file_dir, outputName)
    if not os.path.isfile(outputPath):
        df = None
        for f in tqdm(file_list, desc="Aggregating data"):
            if f.endswith(".json"):
                f_path = os.path.join(file_dir, f)
                if df is None:
                    df = filter_data(pd.read_json(f_path))
                else:
                    # print(f_path)
                    d_ = filter_data(pd.read_json(f_path))
                    df = pd.concat([df, d_])
            # endif
        # endfor
        # shuffle data
        df = shuffle(df, random_state=seed).reset_index(drop=True)
        # save data for test
        df.to_json(outputPath)
    else:
        df = pd.read_json(outputPath)
    return df


def fetch_challenge(prefix="5t20ltl", file_dir="../data/random_ltltrace2", outputName="challenge.json"):
    file_list = os.listdir(file_dir)
    file_list = [f for f in file_list if f.startswith(prefix) and f.endswith(outputName)]
    outputName = ("_" + prefix + ".").join(outputName.split("."))
    outputPath = os.path.join(file_dir, outputName)
    if not os.path.isfile(outputPath):
        df = None
        for f in tqdm(file_list, desc="Aggregating data"):
            if f.endswith(".json"):
                f_path = os.path.join(file_dir, f)
                if df is None:
                    df = filter_data(pd.read_json(f_path))
                else:
                    # print(f_path)
                    d_ = filter_data(pd.read_json(f_path))
                    df = pd.concat([df, d_])
            # endif
        # endfor
        # shuffle data
        df = shuffle(df, random_state=seed).reset_index(drop=True)
        # save data for test
        df.to_json(outputPath)
    else:
        df = pd.read_json(outputPath)
    return df

# filter data
def filter_data(df):
    filtered_index = list(range(len(df)))
    # delete unsat data
    # for i in trange(len(df), desc='Filtering'):
    for i in range(len(df)):
        data = df.loc[i]
        if data.get('trace', -1) == -1:
            filtered_index.remove(i)
        elif data['trace'] == "unsat" or data['trace'] is None:
            filtered_index.remove(i)
    # endfor
    return df.loc[filtered_index]

# 统计时态算子
def countTimeOp(dataframe):
    vocab = r"[a-e01]"
    vocab = re.compile(vocab)
    result = {}
    for i in trange(len(dataframe), desc="Counting"):
        data = dataframe.loc[i]
        if data.get('ltl_pre', -1) == -1 or data['trace'] == "unsat" or data['trace'] is None:
            continue
        ltl = data['ltl_pre']
        processed_ltl = vocab.sub("E", ltl)
        result[processed_ltl] = result.get(processed_ltl, 0) + 1
    # endfor
    path = "../data/5t20ltl_test_distribute_map.json"
    f = open(path, 'w')
    import json
    json.dump(result, f)
    print("Done.")

# 功能：将输入的prefix等分并产生相应前缀的src和val
def aggregate_prefixes(prefixes):
    real_prefix = "5t" + prefixes[-1][-2:]
    print(f"Real prefix: {real_prefix}")

    file_dir = "../data/random_ltltrace2"
    df = None
    for prefix in prefixes:
        path = ("_" + prefix + ".").join("shuffled.json".split("."))
        path = os.path.join(file_dir, path)
        if df is None:
            df = pd.read_json(path)
        else:
            df = pd.concat([df, pd.read_json(path)])
    # endfor
    if len(prefixes) > 1:
        frac = 1 / len(prefixes)
        df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    # endfor
    print(f"Remaining data count: {len(df)}.")
    real_train_path = f"../data/raw/{real_prefix}-src-train.txt"
    real_val_path = f"../data/raw/{real_prefix}-src-val.txt"
    trainSet, validateSet, testSet = process_spec(df)
    writeData(trainSet, real_train_path)
    writeData(validateSet, real_val_path)
    print("Done.")


# 功能：读取以{数据集}-{xtx}命名文件夹内的train、val数据集，并且处理成便于训练的raw数据，输出到processed-{数据集}-{xtx}文件夹下
def processRawData(file_dir, frac):
    if not os.path.isdir(file_dir):
        print(f"File dir: {file_dir} does not exist.")
        return
    else:
        print(f"Processing: {file_dir}.")

    rawDir = file_dir + "-raw"
    if not os.path.isdir(rawDir):
        os.makedirs(rawDir)
    train_path = os.path.join(rawDir, "src-train.txt")
    val_path = os.path.join(rawDir, "src-val.txt")
    test_path = os.path.join(rawDir, "src-test.txt")
    file_list = os.listdir(file_dir)
    for f in file_list:
        path = os.path.join(file_dir, f)
        if f.endswith(".json"):
            df = pd.read_json(path)
            if f.endswith("train.json"):
                writeData(df, train_path, frac)
            if f.endswith("val.json"):
                writeData(df, val_path, frac)
            if f.endswith("test.json"):
                writeData(df, test_path, frac)
        # endif
    # endfor


if __name__ == '__main__':
    # data_name = "trp"
    # range_name = "50t65"
    from argparse import ArgumentParser
    parser = ArgumentParser('Prepare datasets')
    parser.add_argument("--dn", type=str, required=True, help="data name")
    parser.add_argument("--rn", type=str, required=True, help="range name")
    parser.add_argument("--frac", type=float, default=1, help="frac of dataset")
    parser.add_argument("--all", action="store_true", help="process all range")
    args = parser.parse_args()
    data_name = args.dn

    frac = args.frac
    if args.all:
        range_names = ["5t20", "20t35", "35t50", "50t65", "65t80"]
        for rn in range_names:
            processRawData(f"../data/{data_name}-{rn}", frac)
    else:
        range_name = args.rn
        processRawData(f"../data/{data_name}-{range_name}", frac)

    # prefixes = ["ltl5t20", "ltl20t35", "ltl35t50", "ltl50t65", "ltl65t80", "ltl80t105"]
    # for prefix in prefixes:
    #     print(f"{prefix}:")
    #     train_path = f"../data/raw/{prefix}-src-train.txt"
    #     val_path = f"../data/raw/{prefix}-src-val.txt"
    #     test_path = f"../data/test/{prefix}-src-test.txt"
    #     # aggregate data
    #     # print("Reading File...")
    #     # df = fetch_challenge(prefix=prefix, outputName="easy.json")
    #     df = fetch_spec(prefix=prefix)
    #     # df = fetch_spec("", "../data/random_data")
    #     # df = fetch_all()
    #     print(f"Remaining data count: {len(df)}.")
    #     # save data for test
    #     # df.to_json(shuffled_path)
    #     #  We split this set into an 80% training set, a 10% validation set, and a 10% test set.
    #     # trainSet, validateSet, testSet = processData(df)
    #     trainSet, validateSet, testSet = process_spec(df)
    #     # countTimeOp(testSet)
    #     writeData(trainSet, train_path)
    #     writeData(validateSet, val_path)
    #     writeData(testSet, test_path)
    #     print("Done.")
    # # endfor
    #
    # _5t20 = ["ltl5t20"]
    # _5t35 = ["ltl5t20", "ltl20t35"]
    # _5t50 = ["ltl5t20", "ltl20t35", "ltl35t50"]
    # aggregate_prefixes(_5t20)
    # aggregate_prefixes(_5t35)
    # aggregate_prefixes(_5t50)
