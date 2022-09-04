ROOT_DIR = ${PWD}

export PYTHONPATH=$(ROOT_DIR)

jupyter:
	pipenv run jupyter lab

train:
	pipen run python dqn/train.py
