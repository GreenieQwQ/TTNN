from datasets import TranslationDataset, TranslationDatasetOnTheFly
from datasets import TokenizedTranslationDataset
from datasets import IndexedInputTargetTranslationDataset, IndexedInputTargetTranslationDatasetOnTheFly, set_dn_rn
from dictionaries import IndexDictionary
from argparse import ArgumentParser
from utils.pipe import shared_tokens_generator, source_tokens_generator, target_tokens_generator
import os

parser = ArgumentParser('Prepare datasets')
# parser.add_argument('--train_source', type=str, default='../data/raw/{prefix}-src-train.txt')
# parser.add_argument('--train_target', type=str, default='../data/raw/{prefix}-tgt-train.txt')
# parser.add_argument('--val_source', type=str, default='../data/raw/{prefix}-src-val.txt')
# parser.add_argument('--val_target', type=str, default='../data/raw/{prefix}-tgt-val.txt')
parser.add_argument('--train_source', type=str, default='../data/{data_name}-{range_name}-raw/src-train.txt')
parser.add_argument('--train_target', type=str, default='../data/{data_name}-{range_name}-raw/tgt-train.txt')
parser.add_argument('--val_source', type=str, default='../data/{data_name}-{range_name}-raw/src-val.txt')
parser.add_argument('--val_target', type=str, default='../data/{data_name}-{range_name}-raw/tgt-val.txt')
# parser.add_argument("--postfix", type=str, required=True)
parser.add_argument('--save_data_dir', type=str, default='../data/processed-{data_name}-{range_name}')
parser.add_argument("--dn", type=str, required=True, help="data name")
# parser.add_argument("--rn", type=str, required=True, help="range name")
parser.add_argument('--share_dictionary', type=bool, default=False)

args = parser.parse_args()

def prepare(data_name, range_name):
    set_dn_rn(data_name, range_name)  # 设置test on the fly的参数
    # save_data_dir = args.save_data_dir + "-" + args.postfix
    print(f"Data name: {data_name}\tRange name: {range_name}")
    save_data_dir = args.save_data_dir.format(data_name=data_name, range_name=range_name)
    train_source = args.train_source.format(data_name=data_name, range_name=range_name)
    train_target = args.train_target.format(data_name=data_name, range_name=range_name)
    val_source = args.val_source.format(data_name=data_name, range_name=range_name)
    val_target = args.val_target.format(data_name=data_name, range_name=range_name)
    if not os.path.isfile(train_source):
        print(f"File dir: {train_source} does not exist.")
        return
    else:
        print(f"Preparing: {train_source}.")

    TranslationDataset.prepare(train_source, train_target, val_source, val_target, save_data_dir)
    translation_dataset = TranslationDataset(save_data_dir, 'train')
    translation_dataset_on_the_fly = TranslationDatasetOnTheFly('train')

    # print(save_data_dir)
    # print(translation_dataset[0])
    # print(translation_dataset_on_the_fly[0])
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
    indexed_translation_dataset_on_the_fly = IndexedInputTargetTranslationDatasetOnTheFly('train', source_dictionary,
                                                                                          target_dictionary)
    assert indexed_translation_dataset[0] == indexed_translation_dataset_on_the_fly[0]

    print('Done datasets preparation.')

    # 注：vocabulary存储的格式为index{\t}token{\t}个数
    # raw将src和tgt合并为一行 src句子{\t}tgt句子
    # indexed将raw转换为index，并且转换为src, input, answer的形式，
    # 其中input为<SOS> + tgt[:-1]，answer为tgt[1:]+ <EOS>


if __name__ == "__main__":
    dn = args.dn
    # rn = args.rn
    range_names = ["5t20", "20t35", "35t50", "50t65", "65t80"]
    for rn in range_names:
        prepare(dn, rn)
