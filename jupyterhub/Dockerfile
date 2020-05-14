FROM docker.io/jupyter/minimal-notebook:latest
# Get the latest image tag at:
# https://hub.docker.com/r/jupyter/minimal-notebook/tags/
# Inspect the Dockerfile at:
# https://github.com/jupyter/docker-stacks/tree/master/minimal-notebook/Dockerfile
# install additional packages...
COPY ./ /home/jovyan
RUN pip install -r /home/jovyan/science-requirements.txt 
