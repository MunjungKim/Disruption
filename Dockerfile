FROM nvcr.io/nvidia/pytorch:23.04-py3
RUN mkdir -p /libs
ADD /libs/node2vec /libs/node2vec
ADD /libs/util /libs/util
WORKDIR /libs
RUN pip install -e node2vec
RUN pip install -e util
RUN pip install snakemake