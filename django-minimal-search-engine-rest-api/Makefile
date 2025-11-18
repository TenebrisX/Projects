.PHONY: download-data clean-data migrate index test run setup lint format fix-lint help

DATA_DIR = data
DATASET_URL = https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz

help:
	@echo "Available commands:"
	@echo "  make download-data  - Download and setup dataset"
	@echo "  make clean-data     - Remove dataset"
	@echo "  make migrate        - Run database migrations"
	@echo "  make index          - Index documents"
	@echo "  make test           - Run tests"
	@echo "  make run            - Run development server"
	@echo "  make setup          - Full setup (download + index)"
	@echo "  make lint           - Run code linting"
	@echo "  make format         - Format code using black and isort"
	@echo "  make fix-lint       - Automatically fix common linting issues"
	@echo "  make help           - Show this help"

download-data:
	@echo "Downloading NFCorpus dataset..."
	@mkdir -p $(DATA_DIR)
	@cd $(DATA_DIR) && \
	wget -q --show-progress $(DATASET_URL) && \
	tar -xzf nfcorpus.tar.gz && \
	mv nfcorpus/train.docs . && \
	mv nfcorpus/train.titles.queries . && \
	mv nfcorpus/train.3-2-1.qrel . && \
	rm nfcorpus.tar.gz && \
	rm -rf nfcorpus
	@echo "Dataset downloaded successfully!"
	@ls -lh $(DATA_DIR)/

clean-data:
	@echo "Removing dataset..."
	@rm -rf $(DATA_DIR)
	@echo "Dataset removed!"

migrate:
	python manage.py migrate

index:
	python manage.py init --clear

test:
	python manage.py test

run:
	python manage.py runserver

setup: download-data clean-data migrate index
	@echo "Setup completed successfully! Run 'make run' to start the server."

lint:
	@echo "Running flake8..."
	@flake8 . --count --show-source --statistics

format:
	@echo "Formatting code..."
	@black .
	@isort .
	@echo "Code formatted!"

fix-lint: format
	@echo "Fixing common linting issues..."
	@autopep8 --in-place --aggressive --aggressive -r .
	@find . -name "*.py" -exec sed -i 's/[[:space:]]*$//' {} +
	@echo "Run 'make lint' to check remaining issues"