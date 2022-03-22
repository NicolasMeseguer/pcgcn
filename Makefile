default:
	python setup.py install;

run:
	cd pcgcn; python train.py --no-cuda --epochs 4 --nparts 4;

clean:
	python setup.py clean;
