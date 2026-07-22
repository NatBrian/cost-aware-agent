VENV := .venv
PY := $(VENV)/bin/python

# system python3.10 lacks ensurepip; the shared 3.12 build bootstraps via get-pip
venv:
	/mnt/src/python3.12.13/bin/python3 -m venv --without-pip $(VENV)
	curl -sS https://bootstrap.pypa.io/get-pip.py | $(PY) - -q
	$(PY) -m pip install -q -r requirements-cpu.txt

test:
	$(PY) -m pytest tests/ -q

.PHONY: venv test

dry-run:
	$(PY) -m train.grpo_runner --dry-run

data:
	$(PY) scripts/f1_data.py

.PHONY: dry-run data
