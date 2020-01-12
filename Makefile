workflow:
	pipenv run spell workflow --repo repo=. --pip hydra-colorlog --pip hydra-core 'python workflow.py'

deps: Pipfile
	pipenv install --dev
	pipenv lock -r > requirements.txt

test:
	pipenv run py.test

lint:
	pipenv run flake8 workflow.py calbert.py calbert/*.py

.PHONY: test cast lint
