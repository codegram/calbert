workflow:
	poetry run spell workflow --docker codegram/calbert:latest --repo repo=. 'python -m calbert workflow'

deps: pyproject.toml
	poetry install

docker: 
	cp pyproject.toml docker
	cp poetry.lock docker
	docker build -t codegram/calbert ./docker

docker-push:
	docker push codegram/calbert:latest

test:
	poetry run py.test tests

lint:
	poetry run flake8 calbert/*.py

clean:
	rm -fr run train.txt valid.txt tokenizer dataset calbert/__pycache__

.PHONY: test cast lint clean docker docker-push
