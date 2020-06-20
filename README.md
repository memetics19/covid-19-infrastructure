# CoronaWhy Common Research and Data Infrastructure
## What is CoronaWhy?
CoronaWhy.org is a global volunteer organization dedicated to driving actionable insights into significant world issues using industry-leading data science, artificial intelligence and knowledge sharing. 
CoronaWhy was founded during the 2020 COVID-19 crisis, following a White House call to help extract valuable data from more than 50,000 coronavirus-related scholarly articles, dating back decades.
Currently at over 1000 volunteers, CoronaWhy is composed of data scientists, doctors, epidemiologists, students, and various subject matter experts on everything from technology and engineering to communications and program management.

## What has CoronaWhy produced so far?

Read about our [creations](https://github.com/CoronaWhy/covid-19-infrastructure/blob/master/Creations.md) before you start.

## CoronaWhy dashboards

1. [Task-Risk](https://app.powerbi.com/view?r=eyJrIjoiY2E5YjFkZjItN2Q2ZS00MGI5LWFiMWQtZmY0OWRiZTlkNDVmIiwidCI6ImRjMWYwNGY1LWMxZTUtNDQyOS1hODEyLTU3OTNiZTQ1YmY5ZCIsImMiOjEwfQ%3D%3D) helps to identify risk factors that can increase the chance of being infected, or affects the severity or the survival outcome of the infection

2. [Task-Ties](https://app.powerbi.com/view?r=eyJrIjoiOWYwM2Y0OTgtZGE0YS00YjM3LTkwZmYtZTM1NWE5NjJmY2JjIiwidCI6ImRjMWYwNGY1LWMxZTUtNDQyOS1hODEyLTU3OTNiZTQ1YmY5ZCIsImMiOjEwfQ%3D%3D) to explore transmission, incubation and environment stability

3. [Named Entity Recognition](https://app.powerbi.com/view?r=eyJrIjoiMGExNTY3ZjEtMTA3MC00NDYyLTg3YjAtMzZjYTZlMmQ3Mzk3IiwidCI6ImRjMWYwNGY1LWMxZTUtNDQyOS1hODEyLTU3OTNiZTQ1YmY5ZCIsImMiOjEwfQ%3D%3D) across the entire corpus of CORD-19 papers with full text

4. [Match Clinical Trials](https://app.powerbi.com/view?r=eyJrIjoiOGEwYzUwMzctYzJhNS00MjcwLTgzYTktYjQ2ODZmOGM2ZjRkIiwidCI6ImRjMWYwNGY1LWMxZTUtNDQyOS1hODEyLTU3OTNiZTQ1YmY5ZCIsImMiOjEwfQ%3D%3D) allows exploration of the results from the [COVID-19 International Clinical Trials](https://www.kaggle.com/panahi/covid-19-international-clinical-trials) dataset

5. [COVID-19 Literature Visualization](https://app.powerbi.com/view?r=eyJrIjoiODg5ODk5ZGEtYTViMy00ODAzLThiNzMtNWY2MjM5ZWUyNzU3IiwidCI6ImRjMWYwNGY1LWMxZTUtNDQyOS1hODEyLTU3OTNiZTQ1YmY5ZCIsImMiOjEwfQ%3D%3D) helps to explore the data behind the AI-powered literature review

More detailed information about every dashboard published on [Kaggle](https://www.kaggle.com/mikehoney/hyperion/notebook).

## CORD-19 preprocessing pipeline 
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

Follow all updates from our [YouTube](https://www.youtube.com/channel/UCEeuBPsfGE3fceAN3yL5Gig) and [CoronaWhy Github](https://github.com/CoronaWhy/)

# Getting Started with CoronaWhy Common infrastructure
[How to access Elasticsearch and Dataverse, notebook](https://colab.research.google.com/drive/1AO-kBf1MTfqWAUenJJ45vjKsHWBv3men)

[CoronaWhy Elasticsearch Tutorial notebook](https://colab.research.google.com/drive/1dvuzvp2aQsiBiSzv-brh2iA5qbsswRVl#scrollTo=bQ0zEGMsWCJI)

[How to Create Knowledge Graph, notebook](https://colab.research.google.com/drive/1pYVWxG5hXZfkolWe9Q_CZg2hRIfg2Q9u)

[Dataverse Colab Connect, notebook](https://colab.research.google.com/drive/12PmYi8mWILsk4Rci5OqUtavVqk_jQiZH)

[GitHub dataset sync with Dataverse, notebook](https://github.com/mpons/coronawhy-github-dasaset-finder/blob/master/github-links.ipynb)

# CoronaWhy Services
You can connect your notebooks to the number of services listed below, all services coming from CoronaWhy Labs have an experimental status. [Join the fight against COVID-19](https://coronawhy.org) if you want to help us! 

## Data repository

Dataverse deployed as a data service on [https://datasets.coronawhy.org](https://datasets.coronawhy.org)
Dataverse is an open source web application to share, preserve, cite, explore, and analyze research data. It facilitates making data available to others. 

## Elasticsearch

CoronaWhy Elasticsearch has CORD-19 indexes on sentences level and available at [CoronaWhy Search](http://search.coronawhy.org/v9sentences/_search?pretty=true&q=*)

Available indexes:
1. [CORD-19 sentences](http://search.coronawhy.org/v9sentences/_search?pretty=true&q=*) 
2. [CORD-19 sections](http://search.coronawhy.org/cordv7sections/_search?pretty=true&q=*)
3. [GRID Affiliations](http://search.coronawhy.org/grid/_search?pretty=true&q=*) 
4. [MeSH](http://search.coronawhy.org/mesh/_search?pretty=true&q=*)
5. [Geonames](search.coronawhy.org/geonames/_search?pretty=true&q=*)

## MongoDB

MongoDB service deployed on [mongodb.coronawhy.org](mongodb.coronawhy.org) and available from CoronaWhy Labs Virtual Machines. Please contact our administrators if you want to use it.

## Hypothesis

Our Hypothesis annotation service is running on [hypothesis.labs.coronawhy.org](https://hypothesis.labs.coronawhy.org) and allows to manually annotate CORD-19 papers. Please try our [Hypothesis Demo](http://labs.coronawhy.org/hypothesis.html) if you're interested.

## OpenLink Virtuoso triplestore

We are providing [Virtuoso as a service](https://sparql.labs.coronawhy.org) with public [SPARQL Endpoint](https://sparql.labs.coronawhy.org/sparql) that offers an HTTP-based Query Service that operates on Entity Relationship Types (Relations) represented as RDF sentence collections using the SPARQL Query Language. https://virtuoso.openlinksw.com

You can run a simple [SPARQL query](https://sparql.labs.coronawhy.org/sparql?default-graph-uri=&query=select+*+from+%3Chttps%3A%2F%2Fwww.coronawhy.org%3E+where+%7B%3Fs+%3Fp+%3Fo%7D&should-sponge=&format=text%2Fhtml&timeout=0&debug=on) to get some overview of triples from CoronaWhy Knowledge Graph.

## Kibana

Kibana deployed as a community service connected to CoronaWhy Elasticsearch on [https://kibana.labs.coronawhy.org](https://kibana.labs.coronawhy.org)
Allows to visualize Elasticsearch data and navigate the Elastic Stack so you can do anything from tracking query load to understanding the way requests flow through your apps.
https://www.elastic.co/kibana

## BEL

BEL Commons 3.0 available as a service [https://bel.labs.coronawhy.org](https://bel.labs.coronawhy.org)

An environment for curating, validating, and exploring knowledge assemblies encoded in Biological Expression Language (BEL) to support elucidating disease-specific, mechanistic insight.

You can watch the [introduction video](https://www.youtube.com/watch?v=rHhuVBpoKdI&feature=youtu.be) and read [Corona BEL Tutorial](https://docs.google.com/document/d/180gdZMs6AmNi200CTCsMoxGFNPoiE8xxqzG6SCvIQ18/edit#heading=h.kovvpz3y508q) if you want to know more.

### INDRA

INDRA deployed as a service on [https://indra.labs.coronawhy.org/indra](https://indra.labs.coronawhy.org).

INDRA (Integrated Network and Dynamical Reasoning Assembler) generates executable models of pathway dynamics from natural language (using the TRIPS and REACH parsers), and BioPAX and BEL sources (including the Pathway Commons database and NDEx.

You can quickly test the service by running:
```
curl -X POST "https://indra.labs.coronawhy.org/bel/process_pybel_neighborhood" -H "accept: application/json" -H "content-type: application/json" -d "{ \"genes\": [ \"MAP2K1\" ]}" -l -o test_coronawhy_map2k1.json
```

### Geoparser

Geoparser as a service [https://geoparser.labs.coronawhy.org](https://geoparser.labs.coronawhy.org)

The Geoparser is a software tool that can process information from any type of file, extract geographic coordinates, and visualize locations on a map. Users who are interested in seeing a geographical representation of information or data can choose to search for locations using the Geoparser, through a search index or by uploading files from their computer.
https://github.com/nasa-jpl-memex/GeoParser

### Tabula

Tabula allows you to extract data from PDF files into a CSV or Microsoft Excel spreadsheet using a simple, easy-to-use interface. We deployed it as a [CoronaWhy service](http://tabula.labs.coronawhy.org) available for all community members. More information at [Tabula website](https://tabula.technology).

### Teamchatviz

We use [Teamchatviz](https://teamchatviz.labs.coronawhy.org) to explore how communication works in our distributed team and learn how communication shapes culture in CoronaWhy community. https://moovel.github.io/teamchatviz/

### Doccano 
Our Doccano annotation service is running on [doccano.coronawhy.org](https://doccano.coronawhy.org) and allows to manually annotate CORD-19 papers. 



### In progress

We are working on the deployment Neo4j graph database.

# Articles produced by CoronaWhy people
[I’m an AI researcher and here’s how I fight corona](https://medium.com/@arturkiulian/im-an-ai-researcher-and-here-s-how-i-fight-corona-1e0aa8f3e714) by Artur Kiulian


[Exploration of Document Clustering with SPECTER Embeddings](https://medium.com/@beychaner/exploration-of-document-clustering-with-specter-embeddings-7d255f0f7392) by Brandon Eychaner

[COVID-19 Research Papers Geolocation](https://medium.com/swlh/covid-19-research-papers-geolocation-c2d090bf9e06) by Ishan Sharma
