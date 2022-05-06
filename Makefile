install:
	python setup.py install;

run:
	cd pcgcn; python train.py --no-cuda --dataset pubmed --gcn;

clean:
	python setup.py clean;

distclean:
	python setup.py clean; rm -rf build; rm -rf dist; rm -rf pcgcn.egg-info; rm -rf metis;
