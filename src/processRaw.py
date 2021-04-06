import os
import pandas as pd
from sklearn.utils import shuffle
from tqdm import trange, tqdm

seed = 114514
shuffled_path = "../data/random_data/shuffled.json"
train_path = "../data/raw/src-train.txt"
val_path = "../data/raw/src-val.txt"
test_path = "../data/test/src-test.txt"


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


# usage: write the traces and ltls in df to path
def writeData(df, src_path):
    tgt_path = src_path.replace("src", "tgt")
    # containing the automata
    origin_path = src_path.replace("src", "origin")
    df.to_json(origin_path)
    with open(src_path, 'w', encoding='utf-8') as src:
        with open(tgt_path, 'w', encoding='utf-8') as tgt:
            for i in range(len(df)):
                data = df.loc[i]
                src.write(data['ltl'].strip())
                tgt.write(data['trace'].replace("\"", "").replace(",", ";"))
    # endwith

# aggregate data
def fetch_all(file_dir="../data/random_data", outputName="shuffled.json"):
    file_list = os.listdir(file_dir)
    outputPath = os.path.join(file_dir, outputName)
    if not os.path.isfile(outputPath):
        df = None
        for f in tqdm(file_list, desc="Aggregating data"):
            f_path = os.path.join(file_dir, f)
            if df is None:
                df = filter_data(pd.read_json(f_path))
            else:
                d_ = filter_data(pd.read_json(f_path))
                df = pd.concat([df, d_])
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
        elif data['trace'] == "unsat":
            filtered_index.remove(i)
    # endfor
    return df.loc[filtered_index]


if __name__ == '__main__':
    # aggregate data
    df = fetch_all()
    print(f"Remaining data count: {len(df)}.")
    # save data for test
    # df.to_json(shuffled_path)
    #  We split this set into an 80% training set, a 10% validation set, and a 10% test set.
    trainSet, validateSet, testSet = processData(df)
    writeData(trainSet, train_path)
    writeData(validateSet, val_path)
    writeData(testSet, test_path)
