default:
	python setup.py install;

run:
	cd pygcn; python train.py --no-cuda --epochs 4;

clean:
	python setup.py clean;
