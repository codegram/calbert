workflow:
	pipenv run spell workflow --pip-req requirements.txt --repo repo=. 'python -m calbert workflow'

deps: Pipfile
	pipenv install --dev
	pipenv lock -r > requirements.txt

deps-reset:
	pipenv --rm

docker:
	cp requirements.txt docker
	docker build -t codegram/calbert ./docker

docker-push:
	docker push codegram/calbert:latest

test:
	pipenv run py.test

lint:
	pipenv run flake8 calbert/*.py

clean:
	rm -fr run train.txt valid.txt tokenizer dataset calbert/__pycache__

.PHONY: test cast lint clean deps-reset docker docker-push
