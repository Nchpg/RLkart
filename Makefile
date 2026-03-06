clean:
	black .
	isort .

test:
	python -m rl.TestSimulator

train:
	python -m rl.TrainSimulator