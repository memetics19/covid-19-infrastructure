# CoronaWhy Jupyter Notebook Research Infrastructure
Download COVID-19 Open Research Dataset Challenge (CORD-19) from [Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
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
It should produce v12* files in the same folder. File v12_sentences.json contains all extracted entities on sentences level corresponding to CoronaWhy Elasticsearch [collection](http://search.coronawhy.org/v9sentences/_doc/fDLftXEBMjpYvjrLmggF).

# Getting Started with CoronaWhy Common infrastructure
[How to access Elasticsearch and Dataverse (notebook)](https://colab.research.google.com/drive/1AO-kBf1MTfqWAUenJJ45vjKsHWBv3men)
[CoronaWhy Elasticsearch Tutorial (notebook)] https://colab.research.google.com/drive/1dvuzvp2aQsiBiSzv-brh2iA5qbsswRVl#scrollTo=bQ0zEGMsWCJI)

# Articles produced by members of CoronaWhy community
[Exploration of Document Clustering with SPECTER Embeddings](https://medium.com/@beychaner/exploration-of-document-clustering-with-specter-embeddings-7d255f0f7392) by Brandon Eychaner
[COVID-19 Research Papers Geolocation](https://medium.com/swlh/covid-19-research-papers-geolocation-c2d090bf9e06) by Ishan Sharma


# CovonaWhy Services
You can connect your notebooks to the number of services listed below, all services coming from CoronaWhy Labs have an experimental status. [Join the fight against COVID-19](https://coronawhy.org) if you want to help us! 

Data repository
===============

Dataverse deployed as a data service on [https://datasets.coronawhy.org](https://datasets.coronawhy.org)
Dataverse is an open source web application to share, preserve, cite, explore, and analyze research data. It facilitates making data available to others. 

Elasticsearch
===============

CoronaWhy Elasticsearch has CORD-19 indexes on sentences level and available at [CoronaWhy Search](http://search.coronawhy.org/v9sentences/_search?pretty=true&q=*)


MongoDB
===============

MongoDB service deployed on [mongodb.coronawhy.org](mongodb.coronawhy.org) and available from CoronaWhy Labs Virtual Machines. Please contact our administrators if you want to use it.

Hypothesis
===============

Our Hypothesis annotation service is running on [hypothesis.labs.coronawhy.org](https://hypothesis.labs.coronawhy.org) and allows to manually annotate CORD-19 papers. Please try our [Hypothesis Demo](http://labs.coronawhy.org/hypothesis.html) if you're interested.

Kibana
===============

Kibana deployed as a community service connected to CoronaWhy Elasticsearch on [https://kibana.labs.coronawhy.org](https://kibana.labs.coronawhy.org)
Allows to visualize Elasticsearch data and navigate the Elastic Stack so you can do anything from tracking query load to understanding the way requests flow through your apps.
https://www.elastic.co/kibana

BEL
===============
BEL Commons 3.0 available as a service [https://bel.labs.coronawhy.org](https://bel.labs.coronawhy.org)

An environment for curating, validating, and exploring knowledge assemblies encoded in Biological Expression Language (BEL) to support elucidating disease-specific, mechanistic insight.

INDRA
===============

Indra will deployed as a service on [https://labs.coronawhy.org/indra](https://indra.labs.coronawhy.org) (in development).

INDRA (Integrated Network and Dynamical Reasoning Assembler) generates executable models of pathway dynamics from natural language (using the TRIPS and REACH parsers), and BioPAX and BEL sources (including the Pathway Commons database and NDEx.

Geoparser
===============

Geoparser as a service [https://geoparser.labs.coronawhy.org](https://geoparser.labs.coronawhy.org)

The Geoparser is a software tool that can process information from any type of file, extract geographic coordinates, and visualize locations on a map. Users who are interested in seeing a geographical representation of information or data can choose to search for locations using the Geoparser, through a search index or by uploading files from their computer.
https://github.com/nasa-jpl-memex/GeoParser


