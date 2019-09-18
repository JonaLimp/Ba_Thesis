# This must match the Python version in the Docker image.
PYTHON=python3

all:

venv:
	$(PYTHON) -m venv venv
	# TODO always execute this after 'source venv/bin/activate'

install: ./requirements.txt
	pip install --upgrade pip
	pip install --upgrade -r ./requirements.txt

black:
	black code tests setup.py

test:
	$(PYTHON) code/main.py -c configs/simple.yaml
	$(PYTHON) -m flake8 code

clean:
	for dir in ./ tests ; \
	do \
	    find "$$dir" -name '*.pyc' -print0 \
	        -or -name '*.egg-info' -print0 \
	        -or -name '__pycache__' -print0 | \
	        xargs -0 rm -vrf ; \
	done
	rm -f *.log
	rm -rf *.egg-info
	rm -rf logs

distclean:
	git clean -fxd

.PHONY: all install test tox clean distclean
