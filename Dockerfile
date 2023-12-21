FROM jupyter/tensorflow-notebook

COPY requirementsSimEx.txt /requirements.txt

RUN pip install -r /requirements.txt