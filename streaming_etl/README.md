# Streaming in Production Workshop

## Overview
This workshop demonstrates patterns that are useful for building production data pipelines on Databricks. It includes notebooks demonstrating how to load data incrementally using Spark Structured Streaming or Delta Live Tables.

## Structured Streaming notebook
The main notebook to review for using Spark Structured Streaming is [1_flights_structured_streaming](1_structured_streaming/1_flights_structured_streaming). There are some dependencies that should be run first for the workshop setup. In addition, we import Python modules with reusable code which is meant to simplify the logic in the main notebook while still representing realistic transformations if this were a real world data pipeline processing flights data.

## Delta Live Tables notebook and pipeline definition
Run setup notebook from `2_declarative_pipeline/flights_streaming_pipeline/setup/create_pipeline.ipynb`. 

See project code in directory `2_declarative_pipeline/flights_streaming_pipeline`.

## Table optimizations, maintenance, and governance
Explanation and examples of configurations for improved table optimization.

## Alternatvie solutions: dbt
dbt project to build similar tables using SQL based ETL and the dbt framework.

## Contributing and Local Development

- Python: 3.12
- Databricks connect: 16.4
- Compute: Serverless or DBR 16.4 with photon enabled