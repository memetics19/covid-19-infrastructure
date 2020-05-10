# CoronaWhy Jupyter Notebook Research Infrastructure
Download COVID-19 Open Research Dataset Challenge (CORD-19) from Kaggle https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
```
bash ./download_dataset.sh
```
Start Jupyter by executing
```
docker-compose up
```
Jupyter notebook is running on port 8888, test CORD-19 pipeline by running commands:
```
docker cp ./tests covid-19-infrastructure_jupyter_1:/home/jovyan/
docker exec -it covid-19-infrastructure_jupyter_1 /bin/bash
pip install googletrans
cd tests
python ./cord-processing.py
```
It should produce v12* files in the same folder. File v12_sentences.json contains all extracted entities on sentenses level.
