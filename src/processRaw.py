import os
import pandas as pd
from tqdm import trange

seed = 114514

# usage: write the traces and ltls in df to path
def writeData(d, src_path, frac=1):
    # reset index
    df = d.reset_index(drop=True)
    tgt_path = src_path.replace("src", "tgt")
    # containing the automata
    origin_path = src_path.replace("src", "origin").replace(".txt", ".json")
    origin_index = []
    is_val, is_test = "val" in src_path, "test" in src_path
    is_val_test = is_val or is_test
    is_b_o = "balance" in src_path or "one" in src_path
    with open(src_path, 'w', encoding='utf-8') as src:
        with open(tgt_path, 'w', encoding='utf-8') as tgt:
            for i in trange(int(len(df) * frac), desc="writing"):
                data = df.loc[i]
                try:
                    target = data['tgt'].replace("\"", "")
                except KeyError:
                    try:
                        target = data['trace'].replace("\"", "")
                    except KeyError:
                        continue
                # end
                try:
                    source = data['ltl_pre'].strip().replace("->", "I")
                except KeyError:
                    try:
                        source = data['src'].strip().replace("->", "I")
                        # balance和one特化
                        if is_b_o and is_test and source.count("$") == 0 and not args.ta:
                            continue
                    except KeyError:
                        continue
                # end
                if len(target) <= 1024:
                    tgt.write(target + '\n')
                    src.write(source + '\n')
                    # src.write(ltl2prefix(data['ltl'].strip()) + '\n')
                    origin_index.append(i)
                # endif
                if args.cut and is_val_test and len(origin_index) >= int(1e5) * frac:   # 测试集和验证集优上线
                    break
                # endif
    # endwith
    # write origin json
    df.loc[origin_index].reset_index(drop=True).to_json(origin_path)
    print(f"Done writing to {src_path}. Qualified index num: {len(origin_index)}.")
    # endwith


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


# 功能：读取以{数据集}-{xtx}命名文件夹内的train、val数据集，并且处理成便于训练的raw数据，输出到processed-{数据集}-{xtx}文件夹下
def processRawData(file_dir, frac):
    if not os.path.isdir(file_dir):
        print(f"File dir: {file_dir} does not exist.")
        return
    else:
        print(f"Processing: {file_dir}.")

    rawDir = file_dir + "-raw"
    if frac != 1:
        rawDir = rawDir.replace("-5t20-raw", "_8w-5t20-raw")
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
    from argparse import ArgumentParser
    parser = ArgumentParser('Prepare datasets')
    parser.add_argument("--dn", type=str, required=True, help="data name")
    parser.add_argument("--rn", type=str, required=True, help="range name")
    parser.add_argument("--frac", type=float, default=1, help="frac of dataset")
    parser.add_argument("--all", action="store_true", help="process all range")
    parser.add_argument("--ta", action="store_true", help="test all (including proof)")
    parser.add_argument("--cut", action="store_true", help="cut val and test set")
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
