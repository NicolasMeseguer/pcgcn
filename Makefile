default:
	python setup.py install;

run:
	cd pygcn; python train.py;

clean:
	python setup.py clean;
