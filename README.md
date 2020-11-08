# ATLOP
Code for paper [Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling](https://arxiv.org/abs/2010.11304).

If you make use of this code in your work, please kindly cite the following paper:

```bibtex
@article{zhou2020atlop,
  title={Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling},
  author={Zhou, Wenxuan and Huang, Kevin and Ma, Tengyu and Huang, Jing},
  journal={Arxiv},
  year={2020},
  volume={abs/2010.11304}
}
```
## Requirements
* Python (tested on 3.7.4)
* [PyTorch](http://pytorch.org/) (tested on 1.7.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 3.4.0)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* wandb
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The CDR and GDA datasets can be obtained following the intructions in [edge-oriented graph](https://github.com/fenchri/edge-oriented-graph). The expected structure of files is:
```
ATLOP
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |-- gda
 |    |    |-- train.data
 |    |    |-- dev.data
 |    |    |-- test.data
 |-- meta
 |    |-- rel2id.json
```

## Training and Evaluation
### DocRED
Train the BERT model on DocRED with the following command:

```bash
>> sh scripts/run_bert.sh  # for BERT
>> sh scripts/run_roberta.sh  # for RoBERTa
```

The training loss and evaluation scores on the development set are automatically synced to the wandb dashboard.

The program will generate a test file `result.json` in the official evaluation format. You can compress and submit it to Colab for the official test score.

### CDR and GDA
Train CDA and GDA model with the following command:
```bash
>> sh scripts/run_cdr.sh  # for CDR
>> sh scripts/run_gda.sh  # for GDA
```

The training loss and evaluations scores on the development set and test set are automatically syned to the wandb dashboard.

## Saving and Evaluating Models
You can save the trained model by setting the `--save_path` argument in training. The saved model is the model which achieves the best result on the development set. You can evaluate your model by setting the `--load_path` argument, then the code will skip training and evaluate the saved model on benchmarks.
