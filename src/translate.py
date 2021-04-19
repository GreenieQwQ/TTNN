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

dir = "../checkpoints/d_model=128-layers_count=8-heads_count=4-FC_size=512-lr=0.00025-2021_04_18_17_09_21"
parser = ArgumentParser(description='Predict translation')
parser.add_argument('--source', type=str)
parser.add_argument('--config', type=str,
                    default=os.path.join(dir, "config.json"))
parser.add_argument('--checkpoint', type=str,
                    default=os.path.join(dir, "epoch=092-val_loss=0.0903-val_metrics=1.09-0.971.pth"))
# parser.add_argument('--input', required=True,
#                     help='Input for prediction.')
# parser.add_argument('--output', default='../data/prediction/20t35ltl.txt',
#                     help="""Path to output the predictions (each line will
#                         be the decoded sequence""")
parser.add_argument('--beam_size', type=int, default=3)
parser.add_argument('--max_seq_len', type=int, default=100)
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--postfix', type=str, required=True)
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
with open(args.config) as f:
    config = json.load(f)

print(f"Using model: {os.path.basename(args.checkpoint)}")

print('Constructing dictionaries...')
source_dictionary = IndexDictionary.load(config['data_dir'], mode='source')
target_dictionary = IndexDictionary.load(config['data_dir'], mode='target')

print('Building model...')
model = TransformerModel(source_dictionary.vocabulary_size, target_dictionary.vocabulary_size,
                         config['d_model'],
                         config['nhead'],
                         config['nhid'],
                         config['nlayers'])
model.eval()
checkpoint_filepath = args.checkpoint
checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
model.load_state_dict(checkpoint)


def main():
    input_files = ["5t20", "20t35", "35t50", "50t65", "65t80"]
    postfix = args.postfix
    for the_input_file in input_files:
        input_file = "../data/test/" + the_input_file + "ltl-src-test.txt"
        print(f"Translating: {input_file}")
        # 作用：将src进行index
        preprocess = IndexedInputTargetTranslationDataset.preprocess(source_dictionary)
        # 作用：将输出逆index为句子
        postprocess = lambda x: ''.join(
            [token for token in target_dictionary.tokenize_indexes(x) if token != END_TOKEN])
        device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        translator = Translator(
            model=model,
            beam_size=args.beam_size,
            max_seq_len=args.max_seq_len).to(device)

        # 目录
        outputDir = "../data/prediction"
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)

        output_filename = the_input_file + f"ltl-{postfix}.txt"
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
    print('[Info] All Finished.')


if __name__ == "__main__":
    main()
