''' Translate input text with trained model. '''
from model import TransformerModel
from datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary, END_TOKEN

from argparse import ArgumentParser
from translator import Translator
import json
import torch
import os
from dictionaries import special_tokens, PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN
from tqdm import tqdm
from multiprocessing import Process


parser = ArgumentParser(description='Predict translation')
parser.add_argument('--source', type=str)
parser.add_argument("--dn", type=str, required=True, help="data name")
parser.add_argument("--rn", type=str, required=True, help="range name")
parser.add_argument("--tdn", type=str, default="", help="target data name")
parser.add_argument("--epoch", type=str, default="", help="model epoch")
parser.add_argument("--alone", action='store_true')
parser.add_argument('--beam_size', type=int, default=3)
parser.add_argument('--max_seq_len', type=int, default=100)
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--device', type=int, default="0")
parser.add_argument('--bs', type=int, default=128,
                    help='Batch size')
parser.add_argument('--pred_num', type=int, default=int(1e5))
args = parser.parse_args()

config_dir = "../config"
config_path = os.path.join(config_dir, "config.json")
with open(config_path) as f:
    config = json.load(f)

data_name = args.dn
range_name = args.rn
model_dir = "../bestModel"
epoch_postfix = ("_" + args.epoch) if args.epoch != "" else ""
checkpoint_path = os.path.join(model_dir, f"{data_name}-{range_name}{epoch_postfix}.pth")
print(f"Using model: {checkpoint_path}")

dictionary_dir = config['data_dir'] + "-" + data_name + "-" + range_name
# dictionary_dir = config['data_dir'] + "-" + args.postfix
print(f'Constructing dictionaries from {dictionary_dir}...')
source_dictionary = IndexDictionary.load(dictionary_dir, mode='source')
target_dictionary = IndexDictionary.load(dictionary_dir, mode='target')
# 输出目录
outputDir = f"../data/prediction-{data_name}-{range_name}{epoch_postfix}"
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

is_proof_process = True
# 根据dn和rn输出prediction
def predict(dn, rn, device):
    dir_name_format = "../data/{dn}-{rn}-raw"
    dir_name = dir_name_format.format(dn=dn, rn=rn)
    input_path = os.path.join(dir_name, "src-test.txt")
    if not os.path.isfile(input_path):
        print(f"File: {input_path} not exist.")
        return

    output_filename = f"prediction-{dn}-{rn}.txt"
    output_path = os.path.join(outputDir, output_filename)
    if os.path.isfile(output_path):
        print(f"File {output_path} already exists.")
        return

    # 作用：将src进行index
    preprocess = IndexedInputTargetTranslationDataset.preprocess(source_dictionary)
    # 作用：将输出逆index为句子
    postprocess = lambda x: ''.join(
        [token for token in target_dictionary.tokenize_indexes(x) if token != END_TOKEN and token != START_TOKEN and token != PAD_TOKEN])

    print('Building model...')
    model = TransformerModel(source_dictionary.vocabulary_size, target_dictionary.vocabulary_size,
                             config['d_model'],
                             config['nhead'],
                             config['nhid'],
                             config['nlayers'])
    model.eval()
    checkpoint_filepath = checkpoint_path
    checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
    model.load_state_dict(checkpoint)
    translator = Translator(
        model=model,
        beam_size=args.beam_size,
        max_seq_len=args.max_seq_len,
        trg_bos_idx=target_dictionary.token_to_index(START_TOKEN),
        trg_eos_idx=target_dictionary.token_to_index(END_TOKEN)
    ).to(device)

    from utils.pipe import PAD_INDEX
    def pad_src(batch):
        sources_lengths = [len(sources) for sources in batch]
        sources_max_length = max(sources_lengths)
        sources_padded = [sources + [PAD_INDEX] * (sources_max_length - len(sources)) for sources in batch]
        sources_tensor = torch.tensor(sources_padded)
        return sources_tensor
    def process(seq):
        seq = seq.strip()
        def is_proof(name):
            return name.count("balance") > 0 or name.count("one") > 0 or name.count("31") > 0
        if is_proof(data_name) and not is_proof(dn):
            seq += ",$,1"
            global is_proof_process
            if is_proof_process:
                print("processing")
                is_proof_process = False
        return seq

    batch_size = args.bs
    print(f"Output to {output_path}:")
    with open(output_path, 'w', encoding='utf-8') as outFile:
        with open(input_path, 'r', encoding='utf-8') as inFile:
            seqs = []
            for i, seq in tqdm(enumerate(inFile)):
                # if i >= args.pred_num:
                #     print(f"Done translating: num {i}.")
                seq = process(seq)
                src_seq = preprocess(seq)
                seqs.append(src_seq)
                if len(seqs) >= batch_size:
                    pred_seq = translator.translate_sentence(pad_src(seqs).to(device))
                    pred_line = [postprocess(pred) for pred in pred_seq]
                    # print(pred_line)
                    outFile.writelines([p.strip() + '\n' for p in pred_line])
                    seqs.clear()
                # endif
            # endfor
            if seqs:    # last batch
                pred_seq = translator.translate_sentence(pad_src(seqs).to(device))
                pred_line = [postprocess(pred).replace(START_TOKEN, '').replace(END_TOKEN, '') for pred in pred_seq]
                # print(pred_line)
                outFile.writelines([p.strip() + '\n' for p in pred_line])
                seqs.clear()
        # endwith
    # endwith
    print(f'[Info] {input_path} Finished.')


# 功能：对data name里所有的rn进行测试
def main(dn, device):
    range_names = ["5t20", "20t35", "35t50"]
    processes = []
    for rn in range_names:
        p = Process(target=predict, args=(dn, rn, device))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    tdn = args.tdn
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    # 对自己分布进行测试
    if tdn == "":
        main(data_name, device)
    elif tdn.startswith("other"):    # proof特化
        tdns = ["rcf-31", "rf-31", "pf-31"]
        processes = []
        for td in tdns:
            p = Process(target=main, args=(td, device))
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()
    elif tdn == "ps":  # proof特化
        tdn = "spot-31"
        main(tdn, device)
    else:   # 对目标分布测试
        main(tdn, device)
    # endif
