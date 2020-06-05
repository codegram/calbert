deps: pyproject.toml
	poetry install

docker-prepare:
	rm -fr docker/config docker/dist docker/calbert docker/pyproject.toml docker/poetry.lock
	cp pyproject.toml docker
	cp poetry.lock docker
	mkdir -p docker/calbert
	cp calbert/*.py docker/calbert
	cp -r config docker/
	cp -r dist docker/

docker-build: docker-prepare
	docker build -t codegram/calbert ./docker

docker-push:
	docker push codegram/calbert:latest

test:
	poetry run py.test tests

lint:
	poetry run flake8 calbert/*.py tests/*.py

clean:
	rm -fr run calbert/__pycache__

.PHONY: test cast lint clean docker docker-push
