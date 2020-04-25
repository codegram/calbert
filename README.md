# calbert ![](https://github.com/codegram/calbert/workflows/Tests/badge.svg)

`Warning! This is pre-alpha code! Run at your own risk :)`

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

## Training calbert from scratch

All config lives under `config`. There you can control parameters related to training, tokenizing, and everything, and even choose which version of the model to train.

All configuration is overridable, since it's [Hydra](https://hydra.cc) configuration. Check their docs.

### Getting the dataset

You can download the whole dataset and get a small sample to play with locally:

```bash
curl https://traces1.inria.fr/oscar/files/Compressed/ca_dedup.txt.gz -O data.txt.gz
gunzip -c data.txt.gz | head -n 1000 > train.txt
gunzip -c data.txt.gz | tail -n 200 > valid.txt
```

### Training the tokenizer

We're training the tokenizer only on the training set, not the validation set.

```bash
python -m calbert train_tokenizer --input-file train.txt --out-dir tokenizer
```

### Producing the dataset

The dataset is basically a distillation of the raw text data into fixed-length sentences represented by a 4-tuple of tensors `(token_ids, special_tokens_mask, attention_mask, tensor_type_ids)`. Producing these tuples is computationally expensive so we have a separate step for it.

```bash
python -m calbert dataset --train-file train.txt --valid-file valid.txt --tokenizer-dir tokenizer --out-dir dataset
```

### Training the model

```bash
python -m calbert train_model --tokenizer-dir tokenizer --dataset-dir dataset --out-dir model --tensorboard-dir tensorboard
```

Warning, this is really slow! You probably want to run the full thing on GPUs.

### Distributed training

Adjust how many GPUs you have and make sure to tune your batch size in `launch.py`.

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS launch.py
```

### Sharing the model with the world

Once you have a trained model, you can export it to be used as a HuggingFace transformers standard model.

For example, let's imagine you trained a `base-uncased` model and you want to export it.

Remember to override the configuration with your actual model size (`model=base` in this case), but don't worry if you forget --you'll get an error if there is any mismatch.

```bash
python -m calbert export --tokenizer-dir $PWD/calbert-tokenizer-uncased-512 --model-path $PWD/calbert-base-uncased-512.pth --out-dir $PWD/export model=base
```

Now log in to HuggingFace transformers and upload the `export` folder:

```bash
transformers-cli login
transformers-cli upload export
```

### Running tests

```bash
make test
```
