import os
import pandas as pd
from ltlf2dfa.parser.ltlf import LTLfParser, LTLfAnd, LTLfUntil, LTLfNot, LTLfAlways, LTLfAtomic, LTLfNext
from sklearn.utils import shuffle
from tqdm import trange, tqdm

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
def writeData(d, src_path):
    # reset index
    df = d.reset_index(drop=True)
    tgt_path = src_path.replace("src", "tgt")
    # containing the automata
    origin_path = src_path.replace("src", "origin").replace(".txt", ".json")
    origin_index = []
    with open(src_path, 'w', encoding='utf-8') as src:
        with open(tgt_path, 'w', encoding='utf-8') as tgt:
            for i in trange(len(df), desc="writing"):
                data = df.loc[i]
                if isinstance(data['ltl_pre'], str):
                    src.write(data['ltl_pre'].strip().replace("->", "I") + '\n')
                    # src.write(ltl2prefix(data['ltl'].strip()) + '\n')
                    tgt.write(data['trace'].replace("\"", "").replace(",", ";") + '\n')
                    origin_index.append(i)
    # endwith
    # write origin
    df.loc[origin_index].reset_index(drop=True).to_json(origin_path)
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
def fetch_spec(prefix="5t20ltl", file_dir="../data/random_ltltrace", outputName="shuffled.json"):
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


if __name__ == '__main__':
    # prefixes = ["35t50ltl", "50t65ltl", "65t80ltl"]
    # for prefix in prefixes:
    #     # prefix = "5t20ltl"
    #     print(f"{prefix}:")
    #     train_path = f"../data/raw/{prefix}-src-train.txt"
    #     val_path = f"../data/raw/{prefix}-src-val.txt"
    #     test_path = f"../data/test/{prefix}-src-test.txt"
    #     # aggregate data
    #     print("Reading File...")
    #     df = fetch_spec(prefix=prefix)
    #     # df = fetch_spec("", "../data/random_data")
    #     # df = fetch_all()
    #     print(f"Remaining data count: {len(df)}.")
    #     # save data for test
    #     # df.to_json(shuffled_path)
    #     #  We split this set into an 80% training set, a 10% validation set, and a 10% test set.
    #     # trainSet, validateSet, testSet = processData(df)
    #     trainSet, validateSet, testSet = process_spec(df)
    #     writeData(trainSet, train_path)
    #     writeData(validateSet, val_path)
    #     writeData(testSet, test_path)
    #     print("Done.")
    # endfor

    prefixes = ["5t20ltl", "20t35ltl"]
    print(prefixes)
    print("Reading File...")
    file_dir = "../data/random_ltltrace"
    df = None
    for prefix in prefixes:
        path = ("_" + prefix + ".").join("shuffled.json".split("."))
        path = os.path.join(file_dir, path)
        if df is None:
            df = pd.read_json(path)
        else:
            df = pd.concat([df, pd.read_json(path)])
    # endfor
    df = df.sample(frac=0.5, random_state=seed).reset_index(drop=True)
    # endfor
    print(f"Remaining data count: {len(df)}.")
    real_train_path = f"../data/raw/src-train.txt"
    real_val_path = f"../data/raw/src-val.txt"
    trainSet, validateSet, testSet = process_spec(df)
    writeData(trainSet, real_train_path)
    writeData(validateSet, real_val_path)
    print("Done.")