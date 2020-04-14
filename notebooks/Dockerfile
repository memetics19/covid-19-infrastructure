FROM jupyter/datascience-notebook
COPY ./ /home/jovyan/work
WORKDIR /home/jovyan
RUN git clone https://github.com/CoronaWhy/common
RUN pip install -r /home/jovyan/work/requirements.txt
RUN pip install -r /home/jovyan/work/science-requirements.txt
#RUN cat /home/jovyan/common/requirements.txt | cut -f1 -d"#" | perl -ne 'print if s/^(\S+)\=(.+)\=.+$/$1==$2/'| xargs -n 1 pip install #pip install -r /home/jovyan/common/requirements.txt
