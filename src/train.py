from model import TransformerModel
from datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary
from loss_function import TokenCrossEntropyLoss
from metrics import AccuracyMetric
from trainer import EpochSeq2SeqTrainer
from utils.log import get_logger
from utils.pipe import input_target_collate_fn

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from argparse import ArgumentParser
from datetime import datetime
import json
import random

parser = ArgumentParser(description='Train Transformer')
parser.add_argument('--config', type=str, default="../config/config.json")
parser.add_argument('--data_dir', type=str, default='../data/processed')
# parser.add_argument("--postfix", type=str, required=True)
parser.add_argument("--dn", type=str, required=True, help="data name")
parser.add_argument("--rn", type=str, required=True, help="range name")
parser.add_argument('--save_config', type=str, default=None)
parser.add_argument('--save_checkpoint', type=str, default=None)
parser.add_argument('--save_log', type=str, default=None)

parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')

parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--save_every', type=int, default=1)

parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--nhead', type=int, default=2)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--dropout_prob', type=float, default=0.1)

parser.add_argument('--optimizer', type=str, default="Adam", choices=["Noam", "Adam"])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip_grads', action='store_true')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)


def run_trainer(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_name_format = (
        f"data={data_name}-"
        f"range={range_name}-"
        "d_model={d_model}-"
        "layers_count={nlayers}-"
        "heads_count={nhead}-"
        "FC_size={nhid}-"
        "lr={lr}-"
        "{timestamp}"
    )

    run_name = run_name_format.format(**config, timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    logger = get_logger(run_name, save_log=config['save_log'])
    logger.info(f'Run name : {run_name}')
    logger.info(config)

    data_dir = config['data_dir'] + "-" + data_name + "-" + range_name
    logger.info(f'Constructing dictionaries from {data_dir}...')
    source_dictionary = IndexDictionary.load(data_dir, mode='source')
    target_dictionary = IndexDictionary.load(data_dir, mode='target')
    logger.info(f'Source dictionary vocabulary : {source_dictionary.vocabulary_size} tokens')
    logger.info(f'Target dictionary vocabulary : {target_dictionary.vocabulary_size} tokens')

    logger.info('Building model...')
    model = TransformerModel(source_dictionary.vocabulary_size, target_dictionary.vocabulary_size,
                             d_model=config['d_model'],
                             nhead=config['nhead'],
                             nhid=config['nhid'],
                             nlayers=config['nlayers'])
    logger.info(model)
    logger.info('Encoder : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.transformer_encoder.parameters()])))
    logger.info('Decoder : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.transformer_decoder.parameters()])))
    logger.info('Total : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.parameters()])))

    logger.info('Loading datasets...')
    train_dataset = IndexedInputTargetTranslationDataset(
        data_dir=data_dir,
        phase='train')

    val_dataset = IndexedInputTargetTranslationDataset(
        data_dir=data_dir,
        phase='val')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=input_target_collate_fn)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=input_target_collate_fn)

    loss_function = TokenCrossEntropyLoss()
    accuracy_function = AccuracyMetric()
    optimizer = Adam(model.parameters(), lr=config['lr'])

    logger.info('Start training...')
    trainer = EpochSeq2SeqTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function,
        metric_function=accuracy_function,
        optimizer=optimizer,
        logger=logger,
        run_name=run_name,
        save_config=config['save_config'],
        save_checkpoint=config['save_checkpoint'],
        config=config
    )

    trainer.run(config['epochs'])

    return trainer


if __name__ == '__main__':

    args = parser.parse_args()
    data_name = args.dn
    range_name = args.rn
    if args.config is not None:
        with open(args.config, encoding='utf-8') as f:
            config = json.load(f)

        default_config = vars(args)
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
    else:
        config = vars(args)  # convert to dictionary

    run_trainer(config)
