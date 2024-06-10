## INSTALLATION

conda create -n netgan python=3.8.18
conda activate netgan
conda install cudatoolkit=10.0
conda install cudnn=7.3.1

pip install --user nvidia-pyindex
pip install --user nvidia-tensorflow[horovod]
pip install -r requirements.txt


