pip3 install -y dvc[gdrive]
dvc pull nsp-lib-1.5.tar.gz ./data/train.json nsp-result_ya.tqr.gz -r drive
tar xvzf nsp-lib-1.5.tar.gz
tar xvzf nsp-result_ya.tqr.gz