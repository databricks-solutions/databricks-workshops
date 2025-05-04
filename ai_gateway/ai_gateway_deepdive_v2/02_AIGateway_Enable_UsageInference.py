# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Track Endpoint Usage and Log Model Inference 

# COMMAND ----------

# MAGIC %md
# MAGIC Enabling **`Usage Tracking`** and **`Inference Logging`** on Databricks [Mosaic AI Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving) Endpoint is crucial for monitoring and debugging deployed models. It helps in capturing detailed information about the `requests` and `responses` handled by the model endpoints, which can be used to analyze `performance` (e.g. execution duration, status codes, etc.), detect and debug `issues` that arise, and identify areas for improving a model's accuracy and efficiency.
# MAGIC
# MAGIC You can enable [AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints#usage-tracking-table-schemas) **`Usage Tracking`** and **`Inference Logging`** via the **UI** or use the **API** or **`mlflow.deployments`**
# MAGIC
# MAGIC In this notebook we will go over how to do so via **`mlflow.deployments.get_deploy_client(databricks)`** which we also used in our previously defined **`MLflowDeploymentManager`** class. 

# COMMAND ----------

# DBTITLE 1,Run utils.py to access helper functions & set up configs
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Lab User's Workspace | UC | Endpoint Names
## Update to use your own values
CATALOG_NAME = "<your_UC_catalog_name>"
SCHEMA_NAME = "<your_UC_schema_name>"

pt_endpoint_name = f"systemai_endpt_pt_{endpt_name_suffix}" ## endpt_name_suffix variable is derived in ./utils.py
ENDPOINT_NAME = pt_endpoint_name
print("ENDPOINT_NAME: ", ENDPOINT_NAME)

external_endpoint_name = f"external_endpt_{endpt_name_suffix}" ## endpt_name_suffix variable is derived in ./utils.py
EXT_ENDPOINT_NAME = external_endpoint_name
print("EXT_ENDPOINT_NAME: ", EXT_ENDPOINT_NAME)

## Example AzureOpenai API key in Databricks Secrets
# SECRETS_SCOPE = "<secrets_scope>"
# SECRETS_KEY = "AzureOPENAI_API_KEY" # key for AzureOpenAI_API_Token

## if you need to add an External API (e.g. AzureOPENAI_API_KEY) key, you can do so with:
# from databricks.sdk import WorkspaceClient
# w = WorkspaceClient()
# w.secrets.put_secret(scope=SECRETS_SCOPE, key=SECRETS_KEY, string_value='<key_value>')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Update Endpoint `AI Gateway` Config to include Usage Tracking & Inference Logging

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Define **`AI Gateway configuration settings`** for `Usage Tracking` and `Inference Logging`
# MAGIC
# MAGIC At this point we haven't implemented any `ai_gateway` settings on our endpoint configuration. So we will define the settings to enable `usage tracking` and `inference table logging`. 
# MAGIC
# MAGIC Refer to [API documentation for `AI Gateway Configs`](https://docs.databricks.com/api/workspace/servingendpoints/putaigateway)

# COMMAND ----------

# DBTITLE 1,pt_ep_config_aigateway
# Define AI Gateway Configs with usage tracking & inference table logging
pt_ep_config_aigateway={
                        ## Enable usage tracking to track the number of requests
                        "usage_tracking_config": {
                          "enabled": True
                        },
                        ## Enable payload logging to log the request and response
                        "inference_table_config": {
                            "enabled": True,
                            "catalog_name": CATALOG_NAME,
                            "schema_name": SCHEMA_NAME
                        }
                      }

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Leverage our pre-defined `MLflowDeploymentManager()` to execute the endpoint cofiguration updates
# MAGIC
# MAGIC We will instantiate `MLflowDeploymentManager()` class the as our endpoint manager client `ep_manager`

# COMMAND ----------

# DBTITLE 1,MLflowDeploymentManager
# utils.py -- has the class MLflowDeploymentManager defined 

# Instantiate the endpoint manager client
ep_manager = MLflowDeploymentManager()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Push the updated `ai_gateway` configs to the endpoint
# MAGIC
# MAGIC When we make an `update_endpoint_ai_gateway` call with the client manager, our defined AI Gateway Configs with usage tracking & inference table logging gets pushed to the endpoint.

# COMMAND ----------

# DBTITLE 1,update_endpoint_ai_gateway
# Update AI Gateway Configs using the MLflowDeploymentManager
ep_manager.update_endpoint_ai_gateway(pt_endpoint_name, pt_ep_config_aigateway)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4 Review the Serving Endpoint UI 
# MAGIC We now observe that the Endpoint Gateway configurations have the following enabled:   
# MAGIC
# MAGIC - Usage monitoring:
# MAGIC `system.serving.endpoint_usage`
# MAGIC - Dimension table:
# MAGIC `system.serving.served_entities`
# MAGIC - Inference tables:
# MAGIC `{CATALOG_NAME}.{SCHEMA_NAME}.{ENDPOINT_NAME}_payload`
# MAGIC
# MAGIC ![02_enable_usageNinference_logging.png](./imgs/02_enable_usageNinference_logging.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 Get Endpoint Config. Info. with Endpoint Client Manager 
# MAGIC Using the endpoint manager client `ep_manager` is also a great way to quickly check current configs and settings and then to update them where needed. 

# COMMAND ----------

# DBTITLE 1,Review Updated AI GATEWAY config update
## Check our served endpoint details
ep_manager.get_endpoint(pt_endpoint_name) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.5.1 Inspect `ai_gateway` settings within endpoint's config.
# MAGIC We can see that the following has been updated to the endpoint's `ai_gateway` settings with `Usage Tracking` and `Inference Logging` enabled. 
# MAGIC
# MAGIC <br>
# MAGIC     
# MAGIC ```
# MAGIC 'ai_gateway': {'usage_tracking_config': {'enabled': True},
# MAGIC   'inference_table_config': {'catalog_name': 'tko',
# MAGIC   'schema_name': 'ai_gateway_demo',
# MAGIC   'table_name_prefix': 'systemai_endpt_pt_mmt',
# MAGIC   'enabled': True}},
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.5.2 Construct `Payload` name from endpoint config. 
# MAGIC
# MAGIC We can also extract settings for `catalog_name`, `schema_name`, `table_name_prefix` from the `endpoint info.` to construct the `payload` table's `fullname` reference to locate it in the  Unity Catalog. 
# MAGIC
# MAGIC If we try to query the payload before any queries are being sent and logged, it shows up as empty. 

# COMMAND ----------

# DBTITLE 1,Run before Queries are sent - displays empty payload
# Get the endpoint details using client
pt_endpoint_details = ep_manager.get_endpoint(ENDPOINT_NAME)

# Derive the payload table path from the endpoint details
inference_table_config = pt_endpoint_details['ai_gateway'].get("inference_table_config", {})
catalog_name = inference_table_config.get("catalog_name")
schema_name = inference_table_config.get("schema_name")
table_name_prefix = inference_table_config.get("table_name_prefix")

# Construct the full table name
payload_table_fullname = f"{catalog_name}.{schema_name}.{table_name_prefix}_payload"

# Print the extracted values & constructed payload table name
print(f"""Extracted Inference_table_config: \
        \ncatalog_name: {catalog_name}, \
        \nschema_name: {schema_name}, \
        \ntable_name_prefix: {table_name_prefix}, \
        \n\nderived payload_table_fullname: {payload_table_fullname}
        """)
    
# Query the table to check the payload
payload_df = spark.table(payload_table_fullname) 

# Display the payload -- it is empty before any queries are sent to endpoint.
display(payload_df.limit(10)) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.5.3 Prior to Queries being sent to endpoint 
# MAGIC As you would expect, the enabled Inference Logging payload table within Unity Catalog path shows up as empty before any queries are being sent to the model served on the endpoint.   
# MAGIC
# MAGIC ![02_payload_before_anyQueriesSent.png](./imgs/02_payload_before_anyQueriesSent.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Send Queries with Batch Inference using `ai_query()` 
# MAGIC Responses of inference request(s) sent to the model endpoint take a while (`~30--60mins`) to manifest in the Inference logged payload table. In order to access and queiry some loggged requests and responses as part of the lab experience, we shall    
# MAGIC - (1) create some synthetic queries and    
# MAGIC - (2) send these queries by performing [Batch Inference](https://www.databricks.com/blog/introducing-simple-fast-and-scalable-batch-llm-inference-mosaic-ai-model-serving) using `ai_query()` with our provisioned throughput endpoint    
# MAGIC
# MAGIC We will then wait before we check back to explore the payload table later. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Generate Synthetic Queries 
# MAGIC In anticipation of the walking through setting up `Rate Limits` and `Guardrails`, we will create some random queries and inject `sensitive information` into a small percentage of these queries. 
# MAGIC
# MAGIC We will use our _**helper_function**_ **`generate_and_format_queries()`** pre-defined in `./utils.py` to assist here.

# COMMAND ----------

# DBTITLE 1,Generate Random Queries + Inject SensitiveInfo to X% of queries
# Define the required inputs to helper_function

# A "random" list of queries as templates
query_templates = [
                    "What is {topic}?",
                    "How does {topic} work?",
                    "Explain the concept of {topic}.",
                    "Compare {topic1} and {topic2}.",
                    "What are the benefits of {topic}?",
                    "Describe the history of {topic}.",
                    "What are the latest trends for {topic}?"
                  ]

# A "random" list of topics to query
topics = ["artificial intelligence", "quantum computing", "blockchain", 
          "machine learning", "cybersecurity", "data science", "travel", 
          "sci-fi", "a protein", "being kind", "being a foodie"]


sensitive_words = ["pw", "password", "SuperSecretProject"]

messageFlag = '>>>>>>>>>> SENDING ai_query() <<<strInject>>> requests!!! <<<<<<<<<<<'

num_queries = 200
percentage2inject = 10  # N% chance to inject sensitive words -- for this toy example at least 10% is helpful to show the injection

# Call the helper_function 
queries_df = generate_and_format_queries(query_templates, topics, sensitive_words, 
                                         messageFlag, num_queries, percentage2inject)

display(queries_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Inspecting the generated queries, we observe some % of these queries have injected sensitive words.

# COMMAND ----------

# DBTITLE 1,Summary Check on Generated Random Queries
from pyspark.sql import functions as F
from pyspark.sql.window import Window

window_spec = Window.partitionBy()

queries_df_with_percentage = (queries_df.filter(~F.col('query').contains(messageFlag)) 
                              .groupby('strInfo_inject_flag')
                              .count()
                              .orderBy('strInfo_inject_flag')
                              .withColumn('percentage', 
                                          F.round(F.col('count') / F.sum('count').over(window_spec), 2) ##rounding to 2 decimal places
                                          )
                              )

print("Percentage of queries with injected sensitive words")
display(queries_df_with_percentage)

print("Set of queries with injected sensitive words in corresponding modified_query")
display(queries_df.filter(F.col('strInfo_inject_flag') == True))

# COMMAND ----------

# MAGIC %md
# MAGIC Filtering to the queries with the sensitive keywords string injection, we can review where these keywords were inserted in the `modified_query` column. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Perform Batch Inference with `ai_query()`
# MAGIC
# MAGIC We can send our generated synthetic queries to our [provisioned throughput](https://docs.databricks.com/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis.html) endpoint using [Batch Inference with ai_query()](https://docs.databricks.com/en/large-language-models/ai-query-batch-inference) 

# COMMAND ----------

# DBTITLE 1,Batch Inference with ai_query()
from pyspark.sql import functions as F
import time

## create TempView
queries_df.createOrReplaceTempView("queries_df") 

# Define the endpoint and parameters
endpoint_name = pt_endpoint_name
max_tokens = 100 
temperature = 0.7

# Start timing
start_time = time.time()

# Perform the ai_query and add the result as a new column
spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW df_out AS
    SELECT
      *,
      regexp_replace(
        ai_query(
          '{endpoint_name}',
          to_json(named_struct(
            'userAIQ', array(named_struct('prompt', CONCAT(
              'As a helpful and knowledgeable assistant, you will provide short and very succinct responses. Return the response as aiq_response key and corresponding value as strings in a {{"aiq_response":"<string>"}} json. Do not include text before and after the json. >> ',
              modified_query 
            ))),
            'max_tokens', {max_tokens},
            'temperature', {temperature}
            ))),
        '\\\\+', '' -- regexp_replace to remove any number of backslashes in output request json
      ) AS response
    FROM queries_df
""")

# Display the result
display(spark.table("df_out"))

# End timing
end_time = time.time()
# Print execution time
execution_time = end_time - start_time
print(f"Execution time for {endpoint_name}: {execution_time} seconds")


# Create a DataFrame from the temporary view
df_out = spark.sql("SELECT * FROM df_out")


# COMMAND ----------

# DBTITLE 1,to add Markdown
# MAGIC %md
# MAGIC ## 3. Wait for Inference `payload` table to update
# MAGIC  
# MAGIC Sending Queries by [Batch Inferencing](https://docs.databricks.com/aws/en/large-language-models/ai-query-batch-inference) is pretty fast/efficient -- however it will take a while for the request and responses to become logged and manifest in our payload table.
# MAGIC
# MAGIC **_Meanwhile we can explore the sent queries and responses returned to our sparkDF to see how our endpoint model respond to those queries with injected sensitive info..._** 

# COMMAND ----------

## make a copy of output for exploration 
df_out2 = df_out.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1. Check for `sensitive word` `string-injections` in `response` 
# MAGIC
# MAGIC Let's investigate if any of these **_sensitive info_** in the sent queries _**were injected into the response(s)**_ 
# MAGIC
# MAGIC We use a `pandasUDF` to assist with looking for the `sensitive words` in the response(s). 

# COMMAND ----------

# DBTITLE 1,Check if RESPONSE contain sensitive words
from pyspark.sql import functions as F, types as T
import pandas as pd

# sensitive_words = ['pw', 'password', 'SuperSecretProject']

# Define a Pandas UDF to check for sensitive words in the response column
@F.pandas_udf(T.BooleanType())
def contains_sensitive_words_udf(response: pd.Series) -> pd.Series:
    return response.apply(lambda x: any(word in x for word in sensitive_words) if x is not None else False)

df_out2 = df_out2.withColumn("response_contains_sensitive_words", contains_sensitive_words_udf(F.col("response")))

display(df_out2.filter('strInfo_inject_flag == True'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.2 Notice that `some sensitive` info. in queries `got injected into` the `responses`
# MAGIC
# MAGIC We **do** observe some `responses` containing `sensitive words`. Of the queries we sent with known sensitive info. injection, **some yielded `responses` without `sensitive info injection`** from sent `query`, **_however, others yielded `responses` with sensitive info injection_** from the sent `query`. 
# MAGIC
# MAGIC
# MAGIC [**`Impt Note: `** 
# MAGIC _Given some randomness of the `strInfo_inject_flag` and `inject_stringInfo` the numbers you observe in the frequency counts will likely differ._]

# COMMAND ----------

# DBTITLE 1,Get Counts / %
pd_out2 = df_out2.filter(F.col('strInfo_inject_flag')==True).toPandas().groupby('response_contains_sensitive_words')[["response"]].agg('count')
pd_out2['percentage'] = pd_out2['response']/pd_out2['response'].sum()*100
pd_out2

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 `Prompt-injection` can occur without `Guardrails`
# MAGIC **Critically, it is worth highlighting** that _**if **`Guardrails`** are NOT implemented**_ -- _**which we have not yet done so**_ -- it is possible to "coax" the responses to include such sensitive information. Such **_`prompt-injection` is a significant concern in AI and machine learning systems, particularly when dealing with sensitive information_**, with serious consequences pertaining to Data Leakage, Security Risks, Trust and Compliance, as well as Model Integrity.
# MAGIC
# MAGIC We will take a look later **how we can mitigate such behaviour, by implementing `Guardrails` to our `Serving Endpoint(s)`**. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. When recently logged `requests` and `responses` manifest in the `payload`
# MAGIC
# MAGIC After some time (`~30+ mins`), we can take a look at the payload. 
# MAGIC
# MAGIC As we are sending directly from the Databricks platform/notebook so `client_request_id` (e.g. such as queries sent from Web Browsers, Mobile Apps, Desktop Apps, API Clients, IoT Devices, Command-Line Tools, Business Apps, Game Clients etc.) is `null`.
# MAGIC
# MAGIC NB: the payload updates are asynchronous and may take a while to further populate. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Query the `payload` table
# MAGIC
# MAGIC Notice that each `request`  is associated with a `request_time`,`databricks_request_id` and corresponding  `status_code` for the request as well as the `served_entity_id` performing the request. In addition, the columns for `requests` and `responses` consist of values of nested jsons that require parsing.  

# COMMAND ----------

# DBTITLE 1,Query the inference table
print(f'Count of logged requests and responses so far for endpoint: {pt_endpoint_name}')
spark.sql(
    f"""    
    SELECT count(*) over() as total_count
    from {CATALOG_NAME}.{SCHEMA_NAME}.`{ENDPOINT_NAME}_payload`
    Limit 1
    """
).display()

print(f'Show Last 10 rows of logged requests and responses for: {CATALOG_NAME}.{SCHEMA_NAME}.{ENDPOINT_NAME}_payload')
spark.sql(
    f"""
    select *
    from {CATALOG_NAME}.{SCHEMA_NAME}.{ENDPOINT_NAME}_payload  
    sort by request_time desc  
    limit 10
    """
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Extract fields from nested `request` and `response` jsons 
# MAGIC
# MAGIC We can further parse out the `nested json` in the `request` and `response` columns. We have a _**helper-function**_ **`parse_payload()`**
# MAGIC pre-defined in `./utils.py` that we can call. 

# COMMAND ----------

from pyspark.sql import functions as F, types as T

payload_tablename = f"{CATALOG_NAME}.{SCHEMA_NAME}.`{ENDPOINT_NAME}_payload`"

request_date = F.current_date() ## update to a different date where queries were sent if needed

# messageFlag = '>>>>>>>>>> SENDING ai_query() <<<strInject>>> requests!!! <<<<<<<<<<<'
filter_string = messageFlag ## this helps us locate the set of requests i.e. Batch Inference with ai_query() we want to examine

payload_parsed = parse_payload(payload_tablename, request_date, filter_string)

display(payload_parsed.sort('request_time', ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 NB: The `entity_name` for response field `model` in `system.ai` Foundational Model Endpoint `payload` is `empty`.
# MAGIC
# MAGIC We observe that for served `system.ai Foundational Models entities` , there is **no** corresponding served **`entity_name`** information in the parsed `response` json field for **`model`**.    
# MAGIC
# MAGIC However, we can `left-join` the parsed `reponse payload` with **`system.serving.served_entities`** on **`served_entity_id`** to extract the corresponding **`entity_name`** and impute the missing values for the `model` field in the parsed response payload with the corresponding `entity_name` for the served FOUNDATIONAL model(s).  
# MAGIC
# MAGIC [**`Impt Note`:** _For **`external`** Foundational Models, this extra step to join `payload` table with `system.serving.served_entities` table is not necessary as the **`model`** field within the response json is updated with the corresponding mapped **`model's entity_name`** for each configuration update._]

# COMMAND ----------

# DBTITLE 1,L-join on served_entity_id to retrieve model entity_name
import pyspark.sql.functions as F

# Load the served_entities table into a DataFrame
df_served_entities = spark.table("system.serving.served_entities") #spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.served_entities")

# Join the transformed payload DataFrame with the served_entities DataFrame
payload_parsed2 = (payload_parsed
                      .join(
                            df_served_entities,
                            on="served_entity_id",
                            how="left"
                            )
                      .select(
                              payload_parsed["*"],
                              df_served_entities["entity_name"],
                              df_served_entities["endpoint_config_version"],
                              df_served_entities["entity_type"]
                             )
                      .withColumn('model', F.col('entity_name')) ## impute missing 'model' key value with entity_name
                      .select('request_time',
                              'served_entity_id',  
                              'status_code',
                              'entity_name', ## alternatively one could simply use this directly after the join
                              'endpoint_config_version',
                              'entity_type',
                              'request_messages_user_query',
                              'id',
                              'object',
                              'created',
                              'model',
                              'role',
                              'response_messages',
                              'prompt_tokens',
                              'completion_tokens',
                              'total_tokens'
                             )
                      ) 

# Display the result
display(payload_parsed2.sort('request_time', ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.3.1 `served_entity_id` : `entity_name` mapping
# MAGIC
# MAGIC Now we have the mapped `entity_name` corresponding to our `served_entity_id` on our endpoint serving our `system.ai` model: `system.ai.llama_v3_3_70b_instruct`
# MAGIC
# MAGIC In the later section on Model Comparison (`A/B Testing`) we will illustrate how this **`served_entities_id : {entity_name or model_name}`**  mapping will come to be useful to denote which **`endpoint's served_entity`** was `routed` to which `query`. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. NEXT 
# MAGIC We will look at **`Endpoint Usage`** and how to apply **`Query Rate Limits`** where endpoint usage management can be applied for cost control. 
