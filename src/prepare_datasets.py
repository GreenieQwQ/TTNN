from datasets import TranslationDataset, TranslationDatasetOnTheFly
from datasets import TokenizedTranslationDataset
from datasets import IndexedInputTargetTranslationDataset, IndexedInputTargetTranslationDatasetOnTheFly
from dictionaries import IndexDictionary
from argparse import ArgumentParser
from utils.pipe import shared_tokens_generator, source_tokens_generator, target_tokens_generator

parser = ArgumentParser('Prepare datasets')
parser.add_argument('--train_source', type=str, default='../data/raw/{prefix}-src-train.txt')
parser.add_argument('--train_target', type=str, default='../data/raw/{prefix}-tgt-train.txt')
parser.add_argument('--val_source', type=str, default='../data/raw/{prefix}-src-val.txt')
parser.add_argument('--val_target', type=str, default='../data/raw/{prefix}-tgt-val.txt')
parser.add_argument("--postfix", type=str, required=True)
parser.add_argument('--save_data_dir', type=str, default='../data/processed')
parser.add_argument('--share_dictionary', type=bool, default=False)

args = parser.parse_args()
save_data_dir = args.save_data_dir + "-" + args.postfix
args.train_source = args.train_source.format(prefix=args.postfix)
args.train_target = args.train_target.format(prefix=args.postfix)
args.val_source = args.val_source.format(prefix=args.postfix)
args.val_target = args.val_target.format(prefix=args.postfix)
TranslationDataset.prepare(args.train_source, args.train_target, args.val_source, args.val_target, save_data_dir)
translation_dataset = TranslationDataset(save_data_dir, 'train')
translation_dataset_on_the_fly = TranslationDatasetOnTheFly('train', args.postfix)
assert translation_dataset[0] == translation_dataset_on_the_fly[0]

tokenized_dataset = TokenizedTranslationDataset(save_data_dir, 'train')

if args.share_dictionary:
    source_generator = shared_tokens_generator(tokenized_dataset)
    source_dictionary = IndexDictionary(source_generator, mode='source')
    target_generator = shared_tokens_generator(tokenized_dataset)
    target_dictionary = IndexDictionary(target_generator, mode='target')

    source_dictionary.save(save_data_dir)
    target_dictionary.save(save_data_dir)
else:
    source_generator = source_tokens_generator(tokenized_dataset)
    source_dictionary = IndexDictionary(source_generator, mode='source')
    target_generator = target_tokens_generator(tokenized_dataset)
    target_dictionary = IndexDictionary(target_generator, mode='target')

    source_dictionary.save(save_data_dir)
    target_dictionary.save(save_data_dir)

source_dictionary = IndexDictionary.load(save_data_dir, mode='source')
target_dictionary = IndexDictionary.load(save_data_dir, mode='target')

IndexedInputTargetTranslationDataset.prepare(save_data_dir, source_dictionary, target_dictionary)
indexed_translation_dataset = IndexedInputTargetTranslationDataset(save_data_dir, 'train')
indexed_translation_dataset_on_the_fly = IndexedInputTargetTranslationDatasetOnTheFly('train', source_dictionary, target_dictionary, args.postfix)
assert indexed_translation_dataset[0] == indexed_translation_dataset_on_the_fly[0]

print('Done datasets preparation.')

# 注：vocabulary存储的格式为index{\t}token{\t}个数
# raw将src和tgt合并为一行 src句子{\t}tgt句子
# indexed将raw转换为index，并且转换为src, input, answer的形式，
# 其中input为<SOS> + tgt[:-1]，answer为tgt[1:]+ <EOS>
