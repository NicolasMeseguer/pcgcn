# Installs the PCGCN module
install:
	python setup.py install;

# Installs PaRMAT
installrmat:
	cd PaRMAT/Release; make;

# Removes PCGCN module
clean:
	python setup.py clean;

# Removes PCGCN module + directories generated
distclean:
	python setup.py clean; rm -rf build; rm -rf dist; rm -rf pcgcn.egg-info; rm -rf metis;

# Removes PaRMAT
uninstallrmat:
	cd PaRMAT/Release; make clean;

# Executes PCGCN model
run:
	cd pcgcn; python train.py --no-cuda --no-epochs --rmat Blue_Wombat --epochs 2 --gcn;

runtest:
	cd pcgcn; python train.py --no-cuda --no-epochs --rmat Magenta_Spoonbill --epochs 2 --partition metis --nparts 2; python train.py --no-cuda --no-epochs --rmat Magenta_Spoonbill --epochs 2 --partition metis --nparts 4; python train.py --no-cuda --no-epochs --rmat Magenta_Spoonbill --epochs 2 --partition metis --nparts 8; python train.py --no-cuda --no-epochs --rmat Magenta_Spoonbill --epochs 2 --partition metis --nparts 16;