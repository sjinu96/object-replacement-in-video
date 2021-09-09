cd ./lib/resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user

cd ../channelnorm_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user

cd ../correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user

cd ../../pretrained_models
python3 download_flownet2.py

