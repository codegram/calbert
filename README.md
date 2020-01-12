# calbert

A Catalan ALBERT (A Lite BERT), Google's take on self-supervised learning of language representations, trained on the Catalan dataset of [Inria's OSCAR](https://traces1.inria.fr/oscar/) multilingual corpus (729 million unique Catalan words).

## Credits

This is part of the applied research we do at [Codegram](https://codegram.com) (who is to thank for the time and the compute!).

This would have been a ton of pain to build without [Huggingface](http://huggingface.co)'s powerful [transformers](http://github.com/huggingface/transformers) and [tokenizers](http://github.com/huggingface/tokenizers) libraries. Thank you for making NLP actually nice to work with!

Also, thanks to Google Research for creating and open-sourcing [ALBERT](https://github.com/google-research/ALBERT) in the first place.

## Development

### Running tests

```bash
make test
```

### Getting the dataset

You can download the whole dataset and get a small sample to play with locally:

```bash
curl https://traces1.inria.fr/oscar/files/Compressed/ca_dedup.txt.gz -O data.txt.gz
gunzip -c data.txt.gz | head -n 1000 > train.txt
gunzip -c data.txt.gz | tail -n 200 > valid.txt
```

### Configuration

All config lives under `config`. There you can control parameters related to training, tokenizing, and everything, and even choose which version of the model to train.

All configuration is overridable, since it's [Hydra](https://cli.dev) configuration. Check their docs.

### Training the tokenizer

```bash
python calbert.py tokenizer --input-file train.txt --out-dir tokenizer/
```

### Training the model

You need to provide absolute paths with `$PWD` since these are Hydra runs and they don't run on the current directory.

```bash
python calbert.py train --tokenizer-dir $PWD/tokenizer --train-file $PWD/train.txt --eval-file $PWD/valid.txt --out-dir model
```

Warning, this is really slow! You probably want to run the full thing on GPUs. [Spell](https://spell.run) is our platform of choice.

### Running the whole workflow on [Spell](https://spell.run)

```bash
make workflow
```
