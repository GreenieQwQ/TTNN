# TTNN

TEACHING TEMPORAL LOGICS TO NEURAL NETWORKS

## Requirements
- Python 3.6+
- [PyTorch 4.1+](http://pytorch.org/)
- [NumPy](http://www.numpy.org/)
- [tqdm](https://github.com/tqdm/tqdm)

## Usage

### Prepare datasets
This repo comes with example data in `data/` directory. To begin, you will need to prepare datasets with given data as follows:
```
$ python prepare_datasets.py
```

The data consists of parallel source (src) and target (tgt) data for training and validation.
A data file contains one sentence per line with tokens separated by a space.
Below are the provided example data files.

- `src-train.txt`
- `tgt-train.txt`
- `src-val.txt`
- `tgt-val.txt`
  
The test file is as follows.
- `test.txt`

### Predict
To predict a satisfying trace from the source postfix ltl formula:
```
$ python translate.py --config=../checkpoints/config.json --checkpoint=../checkpoints/bestModel.pth --input=../data/test/test.txt
```

It will give you prediction of the given formula in input path and dump the output at the output path.

### Evaluate

To evaluate the prediction with the ground truth and the source ltl formula:
```
$ python evaluate.py --pred=../data/prediction/pred.txt --src=../data/test/test.txt --gd=../data/raw/tgt-train.txt
```

It will output the syntactic accuracy and semantic accuracy.

## References

Parts of code/scripts are borrowed/inspired from:

- https://github.com/pytorch/examples/tree/master/word_language_model

- https://github.com/dreamgonfly/transformer-pytorch

- https://github.com/jadore801120/attention-is-all-you-need-pytorch
