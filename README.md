# Enhancing Neural Temporal Reasoning with Proofs

## Requirements
- Python 3.6+
- [PyTorch 1.7+](http://pytorch.org/)
- [NumPy](http://www.numpy.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [transformers](https://github.com/huggingface/transformers)

## Usage

### Prepare datasets
To begin, you will need to prepare datasets with data in `data/{dataset name}-{range}` directory as follows:
```
$ python processRaw.py --dn={dataset name} --rn={range}
$ python prepare_datasets.py --dn={dataset name} --rn={range}
```
The data directory consists of json files end with "train.json", "val.json" and "test.json". An example element is as follows:
```
{
    "src": "X0,{;c;c},0",
    "tgt": "0,{c;c;},0#@,@,@"
}
```

### Train
To train the model with the specified dataset and range with args in `cofig/config.json`:
```
$  python train --dn={dataset name} --rn={range}
```
for more notification about the arguments, simply consult
```
$  python train -h
```

### Translate
To predict a satisfying trace from the ltl formulas in `test.json` using the best model in `bestModel/` (which is retrieved by copying the best model specified in training log and renamed as `{dataset name}-{range}.pth`):
```
$ python translate.py --dn={dataset name} --rn={range} --tdn={target dataset name}
```

It will give you the prediction of all range in target dataset and output the prediction at `data/preiction-{dataset name}-{range}`.

### Evaluate

To evaluate the prediction produced by the model mentioned above:
```
$ python evaluate.py --dn={dataset name} --rn={range} --tdn={target dataset name}
```

It will output the syntactic accuracy and semantic accuracy of prediction at all range of target dataset.

## References

Parts of code/scripts are borrowed/inspired from:

- https://github.com/pytorch/examples/tree/master/word_language_model

- https://github.com/dreamgonfly/transformer-pytorch

- https://github.com/jadore801120/attention-is-all-you-need-pytorch
