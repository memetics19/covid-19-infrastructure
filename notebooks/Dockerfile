FROM docker.io/coronawhy/jupyter-toolbox:latest
COPY ./ /home/jovyan/work
USER root
WORKDIR /home/jovyan
# Downloading gcloud package
#RUN curl -sSL https://sdk.cloud.google.com | bash

# Adding the package path to local
#RUN mkdir /usr/local/gcloud
#RUN mv /home/jovyan/google-cloud-sdk /usr/local/gcloud/google-cloud-sdk/
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

#RUN git clone https://github.com/CoronaWhy/common
#RUN pip install -r /home/jovyan/work/requirements.txt
#RUN pip install -r /home/jovyan/work/science-requirements.txt
#RUN pip install -r /home/jovyan/work/allenai-requirements.txt 
#RUN cat /home/jovyan/common/requirements.txt | cut -f1 -d"#" | perl -ne 'print if s/^(\S+)\=(.+)\=.+$/$1==$2/'| xargs -n 1 pip install #pip install -r /home/jovyan/common/requirements.txt
