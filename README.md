# covid-19-infrastructure
Jupyter Notebook Research Infrastructure
```
docker-compose up
```
Jupyter notebook is running on port 8888

Test CORD-19 pipeline by running commands:
```
docker cp ./tests covid-19-infrastructure_jupyter_1:/home/jovyan/
docker exec -it covid-19-infrastructure_jupyter_1 /bin/bash
pip install googletrans
cd tests
python ./cord-processing.py
```
It should produce v12* files in the same folder
