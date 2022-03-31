install:
	python setup.py install;

run:
	cd pcgcn; python train.py --no-cuda --epochs 4 --nparts 4 --metis;

clean:
	python setup.py clean;

distclean:
	python setup.py clean; rm -rf build; rm -rf dist; rm -rf pcgcn.egg-info; rm -rf metis;
