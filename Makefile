default:
	python setup.py install;

run:
	cd pcgcn; python train.py --no-cuda --epochs 4 --nparts 4;

clean:
	python setup.py clean;

fullclean:
	python setup.py clean; rm -rf build; rm -rf dist; rm -rf pcgcn.egg-info;
