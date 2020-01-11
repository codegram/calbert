cast:
	pipenv run spell workflow --repo repo=. --pip hydra-colorlog --pip hydra-core 'python workflow.py data.subset=True pretraining.train_batch_size=64 pretraining.num_train_steps=2000 pretraining.num_warmup_steps=150'

deps: Pipfile
	pipenv install --dev
	pipenv lock -r > requirements.txt

test:
	pipenv run py.test

lint:
	pipenv run flake8 workflow.py calbert.py calbert/*.py

.PHONY: test cast lint
