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
runpcgcn:
	cd pcgcn; python train.py --no-cuda --rmat Magenta_Spoonbill --epochs 4 --partition metis --nparts 4;

rungcn:
	cd pcgcn; python train.py --no-cuda --rmat Magenta_Spoonbill --epochs 4 --gcn;

runtest:
	echo "Here you can submit the python script several times, and output the exit to a txt file..."