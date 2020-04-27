deps: pyproject.toml
	poetry install

docker: 
	cp pyproject.toml docker
	cp poetry.lock docker
	docker build -t codegram/calbert -t codegram/calbert:gpu -f docker/Dockerfile.gpu ./docker
	docker build -t codegram/calbert:cpu -f docker/Dockerfile.cpu ./docker

docker-push:
	docker push codegram/calbert:latest
	docker push codegram/calbert:gpu
	docker push codegram/calbert:cpu

test:
	poetry run py.test tests

lint:
	poetry run flake8 calbert/*.py

clean:
	rm -fr run train.txt valid.txt tokenizer dataset calbert/__pycache__

.PHONY: test cast lint clean docker docker-push
