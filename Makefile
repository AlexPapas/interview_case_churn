environment:
	conda env create -f environment.yml --force

jupyter:
	cd ./notebooks; jupyter-notebook

test:
	pytest

format:
	black .
