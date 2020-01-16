workflow:
	pipenv run spell workflow --repo repo=. --pip hydra-colorlog --pip hydra-core 'python -m calbert workflow'

deps: Pipfile
	pipenv install --dev
	pipenv lock -r > requirements.txt

test:
	pipenv run py.test

lint:
	pipenv run flake8 calbert/*.py

.PHONY: test cast lint
