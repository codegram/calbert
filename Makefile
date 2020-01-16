workflow:
	pipenv run spell workflow --pip-req requirements.txt --repo repo=. 'python -m calbert workflow'

deps: Pipfile
	pipenv install --dev
	pipenv lock -r > requirements.txt

test:
	pipenv run py.test

lint:
	pipenv run flake8 calbert/*.py

clean:
	rm -fr run train.txt valid.txt tokenizer dataset calbert/__pycache__

.PHONY: test cast lint clean
