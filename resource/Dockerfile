FROM python:3.14-slim

RUN pip install --no-cache-dir \
    dagster \
    dagster-webserver \
    dagster-postgres \
    dagster-duckdb \
    duckdb \
    pandas

ENV DAGSTER_HOME=/opt/dagster/dagster_home
RUN mkdir -p $DAGSTER_HOME /opt/dagster/app/duckdb_data

WORKDIR /opt/dagster/app

COPY . /opt/dagster/app/