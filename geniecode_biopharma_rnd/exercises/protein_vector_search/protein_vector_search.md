# Protein Embedding and Search
---
**Estimated time:** 30 mins

**Level:** Easy

**Objectives:** Demonstrate:
1. ingestion of OAS antibody sequence data, 
2. generation of protein embeddings using the ESMC model serving endpoint (part of Genesis Workbench)
3. building a Vector Search index
4. querying similarities across unique sequences

**Core Concepts:** Data ingestion to Delta, PySpark QC, ESMC embedding endpoint, Vector Search index creation, similarity search

## Step-by-Step Guide

### 1. Data Ingestion & QC
* Genie Code prompt:
  > Ingest OAS data from https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Kim_2020/csv/SRR12326763_1_Heavy_IGHG.csv.gz into my_catalog.proteinsearch as a raw file in Volumes and as a table as oas_raw. Perform simple QC on the sequences with PySpark, then process out the unique sequences as well as unique sequences with at least 3 occurrences.

#### Expected Results
Saved tarball (128MB) to volume, read CSV into Spark dataframe then perform QC to filter down to 26,013 unique sequences (at least 3 counts) which are saved into a delta table `my_catalog.proteinsearch.oas_unique_min3`.

#### Answer key
- Expected [ingestion NB]($./assets/01_OAS_Ingestion_QC)

---
### 2. ESMC Embeddings & Vector Search
* Genie Code prompt:
  > Using the unique table with sequence count at least 3, create a new notebook to generate a table of ESMC embeddings using the ESMC model endpoint, build a Lakebase Vector Search index over them, and perform similarity searches. Create a new Lakebase project and within it, create a database named <my_username>.

#### Expected Results
Saved tarball (128MB) to volume, read CSV into Spark dataframe then perform QC to filter down to 26,013 unique sequences (at least 3 counts) which are saved into a delta table `my_catalog.proteinsearch.oas_unique_min3`.

#### Answer key
- Expected [Lakebase Search NB]($./assets/02_ESMC_Embeddings_Lakebase_Search)


> **Note:** Regular Vector Search (i.e. **AI Search**) takes ~20 mins to build and ~3mins for inference on ~20k sequences (very efficient). 

> **Lakebase Search** has a build time of ~20s.<br>
To use Lakebase Search, you must: 
- be specific in your prompt to Genie Code, and 
- turn on Lakebase Search in the UI of the Lakebase project page under **Settings** (already done).

## Bonus extensions
* Filtered searches
* Protein structure prediction over returned hits
* Build an app to filter/search mAb sequences or display distributions of common features (e.g. CDR3 length) over the full dataset and over returned hits
