ENVIRONMENT := development

OS := $(shell uname -s)

.PHONY: develop
develop: clean-build install-dev-dependencies
	pip install -e .

.PHONY: install-dev-dependencies
install-dev-dependencies: 
	pip install -r requirements.txt

.PHONY: clean-build
clean-build: 
	rm -rf ./build
	rm -rf ./dist