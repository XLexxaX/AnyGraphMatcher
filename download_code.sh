#!/usr/bin/env bash
wget https://github.com/mnick/scikit-kge/archive/master.zip &&\
unzip master.zip &&\
rm master.zip &&\
cd scikit-kge-master &&\
python3 setup.py install && \
cd .. &&\
rm -r scikit-kge-master