# calbert ![](https://github.com/codegram/calbert/workflows/Tests/badge.svg)

A Catalan ALBERT (A Lite BERT), Google's take on self-supervised learning of language representations.

It's trained on a corpus of **19.557.475 sentence pairs** (containing 729 million unique words) extracted from the Catalan subset of [Inria's OSCAR](https://traces1.inria.fr/oscar/) dataset. We use the a validation set of 833.259 sentence pairs to evaluate the model.

You can read the original [ALBERT paper here](https://arxiv.org/pdf/1909.11942.pdf).

## Pre-trained models

They are available at HuggingFace's [Model Hub page](https://huggingface.co/codegram)

| Model                               | Arch.          | Training data          | Play with it                                                              | Visualize it                                                                                                                                                      |
| ----------------------------------- | -------------- | ---------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `codegram` / `calbert-tiny-uncased` | Tiny (uncased) | OSCAR (4.3 GB of text) | [Card on Model Hub](https://huggingface.co/codegram/calbert-tiny-uncased) | [Visualize in exBERT](https://huggingface.co/exbert/?model=codegram/calbert-tiny-uncased&modelKind=bidirectional&sentence=M%27agradaria%20força%20saber-ne%20més) |
| `codegram` / `calbert-base-uncased` | Base (uncased) | OSCAR (4.3 GB of text) | [Card on Model Hub](https://huggingface.co/codegram/calbert-base-uncased) | [Visualize in exBERT](https://huggingface.co/exbert/?model=codegram/calbert-base-uncased&modelKind=bidirectional&sentence=M%27agradaria%20força%20saber-ne%20més) |

## How to use it?

You just need the `transformers` library. Nothing else to clone or install.

To choose which model version to use (`tiny`, or `base`), consider that smaller models are less powerful, but nimbler and less resource-hungry to run.

```bash
pip install transformers
```

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("codegram/calbert-base-uncased")
model = AutoModel.from_pretrained("codegram/calbert-base-uncased")

model.eval() # disable dropout
```

Now onto the two main use cases that you can do.

### Predicting a missing word in a Catalan sentence

This is the simplest use case, yet not the most useful. Still, here it is! Whatever words you want to mask, use the special token `[MASK]` to indicate it. The model will output the most likely candidates for the masked word.

```python
from transformers import pipeline

calbert_fill_mask  = pipeline("fill-mask", model="codegram/calbert-base-uncased", tokenizer="codegram/calbert-base-uncased")
results = calbert_fill_mask("M'agrada [MASK] això")
# results
# [{'sequence': "[CLS] m'agrada molt aixo[SEP]", 'score': 0.614592969417572, 'token': 61},
#  {'sequence': "[CLS] m'agrada moltíssim aixo[SEP]", 'score': 0.06058056280016899, 'token': 4867},
#  {'sequence': "[CLS] m'agrada més aixo[SEP]", 'score': 0.017195818945765495, 'token': 43},
#  {'sequence': "[CLS] m'agrada llegir aixo[SEP]", 'score': 0.016321714967489243, 'token': 684},
#  {'sequence': "[CLS] m'agrada escriure aixo[SEP]", 'score': 0.012185849249362946, 'token': 1306}]
```

### Extracting a feature vector from a Catalan sentence or document

The extracted feature vector can be used to index documents as dense vectors in ElasticSearch for example, and perform similarity searches.

Another use case is _Natural Language Understanding_ --using these vectors as abstract representations of documents/sentences that can be used as input to other downstream models such as classifiers.

Here's how to extract the vectors from a sentence or document:

```python
import torch
# Tokenize in sub-words with SentencePiece
tokenized_sentence = tokenizer.tokenize("M'és una mica igual")
# ['▁m', "'", 'es', '▁una', '▁mica', '▁igual']

# 1-hot encode and add special starting and end tokens
encoded_sentence = tokenizer.encode(tokenized_sentence)
# [2, 109, 7, 71, 36, 371, 1103, 3]
# NB: Can be done in one step : tokenize.encode("M'és una mica igual")

# Feed tokens to Calbert as a torch tensor (batch dim 1)
encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
embeddings, _ = model(encoded_sentence)
embeddings.size()
# torch.Size([1, 8, 768])
embeddings.detach()
# tensor([[[-0.0261,  0.1166, -0.1075,  ..., -0.0368,  0.0193,  0.0017],
#          [ 0.1289, -0.2252,  0.9881,  ..., -0.1353,  0.3534,  0.0734],
#          [-0.0328, -1.2364,  0.9466,  ...,  0.3455,  0.7010, -0.2085],
#          ...,
#          [ 0.0397, -1.0228, -0.2239,  ...,  0.2932,  0.1248,  0.0813],
#          [-0.0261,  0.1165, -0.1074,  ..., -0.0368,  0.0193,  0.0017],
#          [-0.1934, -0.2357, -0.2554,  ...,  0.1831,  0.6085,  0.1421]]])
```

## Credits

This is part of the applied research we do at [Codegram](https://codegram.com) (who is to thank for the time and the compute!).

This would have been a ton of pain to build without [Huggingface](http://huggingface.co)'s powerful [transformers](http://github.com/huggingface/transformers) and [tokenizers](http://github.com/huggingface/tokenizers) libraries. Thank you for making NLP actually nice to work with!

Also, thanks to Google Research for creating and open-sourcing [ALBERT](https://github.com/google-research/ALBERT) in the first place.

## What on earth is an ALBERT

ALBERT is a Language Model, that is, a neural network that can learn sequences with certain structure, such as sentences in natural language (but not only natural language!).

But how do they learn language? Different language models are trained with different **pretext tasks**, namely challenges that you give them so that they can learn how language works. The idea is that in order to get reaosnably good at this one task they must indirectly learn the grammar of the language, and even its semantics and style.

Traditional (also known as **causal**) language models are usually trained with the task of **predicting the next word** in a sequence, like this:

- Input: "the dog was eating very [BLANK]"
- Correct output: "quickly"

However, ALBERT is of another family called **masked language models**. In this family, the pretext task they have to learn is similar, but instead of always predicting the last word in a sequence, some words in the sentence are randomly turned into blanks (or **masked**), like this:

- Input: "the [BLANK] was eating very [BLANK]"
- Correct output: "dog", "quickly"

This task is a little more difficult, and more importantly, requires understanding the context surrounding a blank much better.

### How are those pretext tasks anything more than a pointless waste of electricity

Turns out, once a language model gets really, really good at this rather pointless pretext task, it can be easily repurposed for much more interesting tasks.

Once a language learns grammar and semantics, it can become a very good classifier of sentences, and even whole documents, for example.

If you then teach it to classify tweets or documents into categories (or identify sentiment, or toxicity for example) it no longer sees just a bunch of confusing characters, but rather it's "reading" the document at a much more abstract level, so it can "make sense" of it much more readily. (Note the air quotes, this is not magic but it is probably the closest thing.)

### Why ALBERT in Catalan

Because there are no language models in Catalan! And there's a lot of Catalan text to be processed. (In Catalonia).

## Setup

For dependency management we use [Poetry](https://python-poetry.org) (and Docker of course).

```bash
pip install -U poetry
poetry install
poetry shell
```

The production image to train the model is under `docker/`, and it's called `codegram/calbert`. It contains all the latest dependencies, but no code -- Deepkit will ship the code in every experiment (read on to learn more about Deepkit).

## Dataset and tokenizers

All config lives under `config`. There you can control parameters related to training, tokenizing, and everything, and even choose which version of the model to train.

All configuration is overridable, since it's [Hydra](https://hydra.cc) configuration. Check their docs.

### Getting the dataset

A tiny subset of the dataset lives under `dist/data` so that you can train a small model and do quick experiments locally.

To download the full dataset and automatically split it in training / validation, just run this command:

```bash
python -m calbert download_data --out-dir dataset
```

### Re-training the tokenizers

The pretrained tokenizers are at `dist/tokenizer-{cased,uncased}`. They are trained only on the full training set.

If you want to re-train the tokenizer (by default uncased):

```bash
python -m calbert train_tokenizer --input-file dataset/train.txt --out-dir tokenizer
```

To train the cased one, just override the appropriate Hydra configuration:

```bash
python -m calbert train_tokenizer --input-file dataset/train.txt --out-dir tokenizer vocab.lowercase=False
```

## Training and running experiments

We use [Deepkit](https://deepkit.ai) to run and keep track of experiments. Download it for free for your platform of choice if you'd like to run locally, or check their docs to run against their free community server.

### Training a test model

To make sure everything works, let's train a test model with the actual Docker image in Deepkit:

```bash
deepkit run test.deepkit.yml
```

By default it will train it in your local Deepkit instance, using your CPU. Read [their docs](https://deepkit.ai/documentation/getting-started) to learn how to customize your runs.

### Training on a cluster

Configure a cluster in your local Deepkit with at least one machine with a GPU.

```bash
deepkit run --cluster
```

### Sharing the model with the world

Once you have a trained model, you can export it to be used as a HuggingFace transformers standard model.

For example, let's imagine you trained a `base-uncased` model and you want to export it.

Download the `export` folder from the outputs in your Deepkit run, and run:

```bash
mv export calbert-base-uncased
transformers-cli login
transformers-cli upload export
```

### Running tests

```bash
make test
```
