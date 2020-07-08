# calbert ![](https://github.com/codegram/calbert/workflows/Tests/badge.svg)

A Catalan ALBERT (A Lite BERT), Google's take on self-supervised learning of language representations.

It's trained on a corpus of **19.557.475 sentence pairs** (containing 729 million unique words) extracted from the Catalan subset of [Inria's OSCAR](https://traces1.inria.fr/oscar/) dataset. We use the a validation set of 833.259 sentence pairs to evaluate the model.

You can read the original [ALBERT paper here](https://arxiv.org/pdf/1909.11942.pdf).

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
