.PHONY: clean-temp ruff-format ruff-chec-fix ruff

clean-temp:
	find . -type f -name '._*' -delete

ruff-format:
	ruff format .

ruff-chec-fix:
	ruff check --fix --unsafe-fixes .

ruff:
	ruff check --fix --unsafe-fixes .
	ruff format .
