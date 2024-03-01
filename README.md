# Item

The Instruction Trace Embedding Model (ITEM) repository.

## Installation

Install the dependencies for this project with pip.

```bash
pip install -r requirements.txt
```

## Usage

This repository is essentially a collection of scripts for training and using
the ITEM model. Scripts include informative argparse help messages.

### Preprocessing

First, preprocess the binaries into tokenizable texts.

```bash
python preprocess.py \
    /path/to/sample1 /path/to/sample2 ... \
    -o /path/to/preprocessed/output/
```

Next, train the tokenizer on the preprocessed texts.

```bash
python tokenizer/train.py \
    /path/to/preprocessed/output/ \
    -o /path/to/item.tokenizer.json
```

### Pretraining

Once you have preprocessed your samples and trained a tokenizer on them, you
can then create a pretraining dataset from the preprocessed binaries.

```bash
python dataset/pretraining/process.py \
    /path/to/preprocessed/output/* \
    -t /path/to/item.tokenizer.json \
    -o /path/to/pretraining.dataset
```

You can now begin pretraining.

```bash
python model/pretrain.py \
    /path/to/pretraining.dataset \
    -t /path/to/item.tokenizer.json \
    -o /path/to/model.pretraining.output
```

### Fine-Tuning

After preprocessing your data, training a tokenizer, and pretraining a model
you can now fine tune your model for a similarity task. First, you'll need to
create a fine-tuning dataset from your preprocessed binaries.

```bash
python dataset/contrasting/process.py \
    /path/to/preprocessed/output/* \
    -t /path/to/item.tokenizer.json \
    -o /path/to/contrasting.dataset
```

You can now begin fine-tuning.

```bash
python model/contrast.py \
    /path/to/contrasting.dataset \
    -t /path/to/item.tokenizer.json \
    -m /path/to/model.pretraining.output/epoch-number \
    -o /path/to/model.contrasting.output
```

### Inference

A simple example inference script is included that computes the similarity of
two functions from two given binaries. Once you have a fine-tuned model with
acceptable validation performance, you can use this script to compare
functions.

```bash
python inference.py \
    -t /path/to/item.tokenizer.json \
    -m /path/to/model.contrasting.output/epoch-number \
    /path/to/first-binary:function_name_or_address \
    /path/to/second-binary:function_name_or_address \
```

This script can also be used as a template for integration with other
tools/systems, given a trained model.

## Distribution

DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.

This material is based upon work supported by the Combatant Commands under Air
Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or
recommendations expressed in this material are those of the author(s) and do
not necessarily reflect the views of the Combatant Commands.

© 2023 Massachusetts Institute of Technology

The software is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice,
U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS
252.227-7014 as detailed above. Use of this work other than as specifically
authorized by the U.S. Government may violate any copyrights that exist in this
work.

[MIT License](LICENSE.txt)
