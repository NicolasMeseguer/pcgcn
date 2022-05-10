install:
	python setup.py install;

run:
	cd pcgcn; python train.py --no-cuda --dataset citeseer --epochs 500 --gcn;

runtest:
	echo "Here you can submit the python script several times, and output the exit to a txt file..."

clean:
	python setup.py clean;

distclean:
	python setup.py clean; rm -rf build; rm -rf dist; rm -rf pcgcn.egg-info; rm -rf metis; rm data/graphlaxy/*.graph
