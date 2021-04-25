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


parser = ArgumentParser(description='Predict translation')
parser.add_argument('--source', type=str)
# parser.add_argument('--config', type=str,
#                     default=os.path.join(model_dir, "config.json"))
# parser.add_argument('--checkpoint', type=str,
#                     default=os.path.join(model_dir, model_path))
# parser.add_argument('--input', required=True,
#                     help='Input for prediction.')
# parser.add_argument('--output', default='../data/prediction/20t35ltl.txt',
#                     help="""Path to output the predictions (each line will
#                         be the decoded sequence""")
parser.add_argument('--beam_size', type=int, default=3)
parser.add_argument('--max_seq_len', type=int, default=100)
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--postfix', type=str, required=True)
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--challenge', action="store_true",
                    help='Activate challenge mode.')
# TODO: Translate bpe encoded files
# parser.add_argument('-src', required=True,
#                    help='Source sequence to decode (one line per sequence)')
# parser.add_argument('-vocab', required=True,
#                    help='Source sequence to decode (one line per sequence)')
# TODO: Batch translation
# parser.add_argument('-batch_size', type=int, default=30,
#                    help='Batch size')
# parser.add_argument('-n_best', type=int, default=1,
#                    help="""If verbose is set, will output the n_best
#                    decoded sentences""")

args = parser.parse_args()
postfix = args.postfix
if postfix == "5t20ltl":
    model_dir = "../checkpoints/d_model=128-layers_count=8-heads_count=4-FC_size=512-lr=0.00025-2021_04_17_21_08_25"
    model_path = "epoch=092-val_loss=0.0653-val_metrics=1.07-0.979.pth"
elif postfix == "5t35ltl":
    model_dir = "../checkpoints/d_model=128-layers_count=8-heads_count=4-FC_size=512-lr=0.00025-2021_04_18_17_09_21"
    model_path = "epoch=092-val_loss=0.0903-val_metrics=1.09-0.971.pth"
elif postfix == "5t50ltl":
    model_dir = "../checkpoints/d_model=128-layers_count=8-heads_count=4-FC_size=512-lr=0.00025-2021_04_19_13_13_55"
    model_path = "epoch=086-val_loss=0.108-val_metrics=1.11-0.965.pth"
else:
    model_dir, model_path = None, None

config_path = os.path.join(model_dir, "config.json")
with open(config_path) as f:
    config = json.load(f)

checkpoint_path = os.path.join(model_dir, model_path)
print(f"Using model: {os.path.basename(checkpoint_path)}")

dictionary_dir = config['data_dir'] + "_" + postfix
print(f'Constructing dictionaries from {dictionary_dir}...')
source_dictionary = IndexDictionary.load(dictionary_dir, mode='source')
target_dictionary = IndexDictionary.load(dictionary_dir, mode='target')

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


def main():
    input_files = ["5t20", "20t35", "35t50", "50t65", "65t80"]
    is_challenge = args.challenge
    mode = args.mode
    for the_input_file in input_files:
        input_file = "../data/test/" + the_input_file + (f"ltl-src-{mode}.txt" if is_challenge else "ltl-src-test.txt")
        print(f"Translating: {input_file}")
        # 作用：将src进行index
        preprocess = IndexedInputTargetTranslationDataset.preprocess(source_dictionary)
        # 作用：将输出逆index为句子
        postprocess = lambda x: ''.join(
            [token for token in target_dictionary.tokenize_indexes(x) if token != END_TOKEN])
        device = torch.device('cuda:1' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        translator = Translator(
            model=model,
            beam_size=args.beam_size,
            max_seq_len=args.max_seq_len).to(device)

        # 目录
        outputDir = "../data/prediction"
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)

        output_filename = the_input_file + (f"ltl-{mode}-{postfix}.txt" if is_challenge else f"ltl-{postfix}.txt")
        output_path = os.path.join(outputDir, output_filename)
        print(f"Output to {output_path}:")
        with open(output_path, 'w', encoding='utf-8') as outFile:
            with open(input_file, 'r', encoding='utf-8') as inFile:
                for seq in tqdm(inFile):
                    src_seq = preprocess(seq)
                    pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
                    pred_line = postprocess(pred_seq)
                    pred_line = pred_line.replace(START_TOKEN, '').replace(END_TOKEN, '')
                    # print(pred_line)
                    outFile.write(pred_line.strip() + '\n')
        print('[Info] Finished.')
    if is_challenge:
        print(f'[Info] {mode} Finished.')
    else:
        print('[Info] Random Finished.')


if __name__ == "__main__":
    main()
