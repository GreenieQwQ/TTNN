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

parser = ArgumentParser(description='Predict translation')
parser.add_argument('--source', type=str)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--input', required=True,
                    help='Input for prediction.')
parser.add_argument('--output', default='../data/prediction/pred.txt',
                    help="""Path to output the predictions (each line will
                        be the decoded sequence""")
parser.add_argument('--beam_size', type=int, default=3)
parser.add_argument('--max_seq_len', type=int, default=100)
parser.add_argument('--no_cuda', action='store_true')
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
    '''Main Function'''
    # 作用：将src进行index
    preprocess = IndexedInputTargetTranslationDataset.preprocess(source_dictionary)
    # 作用：将输出逆index为句子
    postprocess = lambda x: ''.join([token for token in target_dictionary.tokenize_indexes(x) if token != END_TOKEN])
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    translator = Translator(
        model=model,
        beam_size=args.beam_size,
        max_seq_len=args.max_seq_len).to(device)

    # 生成目录
    outputDir = os.path.dirname(args.output)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    with open(args.output, 'w', encoding='utf-8') as outFile:
        with open(args.input, 'r', encoding='utf-8') as inFile:
            for seq in inFile:
                src_seq = preprocess(seq)
                pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
                pred_line = postprocess(pred_seq)
                pred_line = pred_line.replace(START_TOKEN, '').replace(END_TOKEN, '')
                # print(pred_line)
                outFile.write(pred_line.strip() + '\n')

    print('[Info] Finished.')


if __name__ == "__main__":
    main()
