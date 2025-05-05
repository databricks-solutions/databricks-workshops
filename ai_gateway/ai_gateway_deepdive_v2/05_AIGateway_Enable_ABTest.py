# Databricks notebook source
# MAGIC %md
# MAGIC # 05 A/B Test | Traffic Routing with `AI-Gateway`
# MAGIC
# MAGIC Model comparison and A/B testing are essential practices in machine learning and AI development for several reasons:
# MAGIC - **Performance Evaluation:** To determine which model performs better in real-world scenarios by comparing metrics such as accuracy, precision, recall, and latency.
# MAGIC - **User Experience:** To ensure that the model provides the best possible user experience by testing different versions and configurations.
# MAGIC - **Optimization:** To identify the most effective model or configuration for specific tasks and user profiles, leading to performance gains.
# MAGIC - **Risk Mitigation:** To minimize the risk of deploying a suboptimal model by thoroughly testing and comparing it against existing models.
# MAGIC
# MAGIC #### Why Use Databricks AI Gateway to Implement It?
# MAGIC Databricks [AI Gateway](https://docs.databricks.com/aws/en/ai-gateway) offers several advantages for implementing model comparison and A/B testing:
# MAGIC - **Centralized Governance:** AI Gateway provides a centralized service for governing and monitoring access to generative AI models and their associated model serving endpoints.
# MAGIC - **Unified Platform:** It unifies the data layer and ML platform, making it possible to track lineage from raw data to production models.
# MAGIC - **Built-in Monitoring:** AI Gateway allows for automatic collection and monitoring of inference tables that contain request and response data for an endpoint.
# MAGIC - **Scalability:** It supports the deployment of multiple model serving endpoints, enabling the setup of separate endpoints for each model to facilitate A/B testing.
# MAGIC - **Integration with External Load Balancers:** For current limitations (see `NOTE` below), AI Gateway can be integrated with external load balancers to distribute traffic and implement custom logic for A/B testing.   
# MAGIC
# MAGIC #### Example Implementation
# MAGIC In this notebook we will show  an example of how you might set up **`A/B test`** using Databricks Model Serving with [**`AI Gateway`**](https://docs.databricks.com/aws/en/ai-gateway) as well as track the traffic routed to their respective `served_entities_ids`.      
# MAGIC
# MAGIC
# MAGIC ---    
# MAGIC
# MAGIC _**`Impt Note:`**_ **`Current Model Serving Endpoint limitations and response json output differences:`** 
# MAGIC - **`No mix model`** (e.g. Databricks-hosted 'internal'/`system.ai` & `external` Azure OpenAI or Anthropic or Claude etc OR custom) types on a single endpoint
# MAGIC   - `Workaround for current limitations regarding traffic splitting or A/B testing across different types of models:` Option to serve each model separately and integrate an external (`AWs/Azure/GC`) load balancer service application with custom logic to distribute the traffic. 
# MAGIC
# MAGIC - **`External model endpoint on Databricks`** have payload `model` info. in response key and also provide automatic mapping of `model` with  `served_entity_id`; this is not the case for Foundational models registered on `system.ai` -- mapping is derived by joining payload information with the `system.serving.served_entities` on `served_entity_id`. (User experience may be affected.) 
# MAGIC

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
# MAGIC ## 1. Endpoint `ai_gateway` Config. Update Process: [`A/BTest`]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Leverage our pre-defined `MLflowDeploymentManager()` 
# MAGIC
# MAGIC We will once again instantiate the **`MLflowDeploymentManager()`** client and use the _**helper-class**_'s functionality to retrieve and execute the endpoint cofiguration updates. 

# COMMAND ----------

# DBTITLE 1,MLflowDeploymentManager
# utils.py -- has the class MLflowDeploymentManager defined 

# Instantiate the manager
ep_manager = MLflowDeploymentManager()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Update Exisiting Endpoint `ai_gateway` Config: `pt_ep_config_aigateway`
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2.1 Review `Endpoint Config` & `ai_gateway` Settings:
# MAGIC
# MAGIC Before we begin, let's briefly review the current `Endpoint Configurations` corresponding `AI Gateway Settings`.

# COMMAND ----------

# DBTITLE 1,RUN Once before updating endpoint &  ai_gateway  ai_gateway configs
ep_info = ep_manager.get_endpoint(pt_endpoint_name)
ep_info

# COMMAND ----------

# MAGIC %md
# MAGIC There are attributes in the `ai_gateway` settings we will want to maintain and some we will will to remove for the A/B Test configuration walk through. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2.2 Define NEW `system.ai` (Foundational) Endpoint Configs & AI Gateway Settings 
# MAGIC
# MAGIC
# MAGIC We will update our `system.ai` `Model Serving Endpoint` to serve another Foundational model and route their query traffic 50/50%
# MAGIC
# MAGIC At the same time we will want to revert the `5 QPM` Rate Limits to the default settings and leverage the model [provisioned throughput](https://docs.databricks.com/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis.html) so that we can send bulk queries as batch inferences more efficiently to demonstrate the routing of queries to the endpoint serving 2 models.  
# MAGIC  
# MAGIC To do so, we need to define a new config for updating our `system.ai` endpoint's current `AI Gateway` settings. 
# MAGIC
# MAGIC We will use our instantiated `MLflowDeploymentManager()` to update both the endpoint `configs` and the `ai_gateway` settings.
# MAGIC
# MAGIC In the following code cells, we show how this is achieved; we provide commented code snippets to highlight what is updated or omitted. 

# COMMAND ----------

# DBTITLE 1,Compare | A/BTest 2 "system.ai" Models
systemai_new_config = {
                        "served_entities": [
                            # model A
                            {
                                "name": "llama_v3_3_70b_instruct-1",
                                "entity_name": "system.ai.llama_v3_3_70b_instruct",
                                "entity_version": "1",
                                "min_provisioned_throughput": 0,
                                "max_provisioned_throughput": 9500,
                                "scale_to_zero_enabled": True,
                            },
                            # model B 
                            {
                                "name": "meta_llama_v3_1_8b_instruct-3",
                                "entity_name": "system.ai.meta_llama_v3_1_8b_instruct",
                                "entity_version": "3",
                                "min_provisioned_throughput": 0,
                                "max_provisioned_throughput": 9500,
                                "scale_to_zero_enabled": True,
                            }
                        ],
                        "traffic_config": {
                            "routes": [
                                {
                                    "served_model_name": "llama_v3_3_70b_instruct-1",
                                    "traffic_percentage": 50
                                },
                                {
                                    "served_model_name": "meta_llama_v3_1_8b_instruct-3",
                                    "traffic_percentage": 50
                                }
                            ]
                        },
                        "tags": [
                            {"key": "owner", "value": "user_info"},
                            {"key": "removeAfter", "value": "2025-03-31"},
                            {"key": "tko-demo", "value": "true"}
                        ]
                    }


aigateway_config = {
                    'usage_tracking_config': {'enabled': True},
                    # ## omit if already enabled 
                    # 'inference_table_config': {'catalog_name': f'{CATALOG_NAME}',
                    #                            'schema_name': f'{SCHEMA_NAME}',
                    #                            'table_name_prefix': f'{pt_endpoint_name}',
                    #                            'enabled': True
                    #                           },
                    # ## remove limits for this example so we can leverage provisioned throughput in the batch inference
                    # 'rate_limits': [{'calls': 5, 'key': 'endpoint', 'renewal_period': 'minute'},
                    #                 {'calls': 5, 'key': 'user', 'renewal_period': 'minute'}
                    #                ],
                    'guardrails': {
                                    'input': {
                                        'safety': False,
                                        'pii_detection': True,
                                        'pii': {'behavior': 'BLOCK'},
                                        'invalid_keywords': ['SuperSecretProject', 'pw', 'password']
                                    },
                                    'output': {
                                                'safety': False,
                                                'pii_detection': True,
                                                'pii': {'behavior': 'BLOCK'}
                                            }
                                  }
                    }

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2.3 Update `system.ai` Endpoint Configs & AI Gateway `Settings with MLflowDeploymentManager()`
# MAGIC
# MAGIC Now we push both the new `endpoint configs` with updated `ai_gateway` `settings` to `endpoint`. 

# COMMAND ----------

# DBTITLE 1,update_endpoint_config
# Update the deployed provisioned throughput endpoint serving system.ai model -- this takes a few minutes to provision

import time
from requests.exceptions import HTTPError

for _ in range(5):
    try:        
        # 1st update the endpoint config
        ep_manager.update_endpoint_config(pt_endpoint_name, systemai_new_config)
         
        # 2nd update the endpoint ai gateway config
        ep_manager.update_endpoint_ai_gateway(pt_endpoint_name, aigateway_config) 

        print('-'*100)       
        # ep_manager.get_endpoint(pt_endpoint_name, print_info=True)

        if is_update_in_progress(ep_manager, pt_endpoint_name, timeout=300):
            print('-'*100)
            ep_manager.get_endpoint(pt_endpoint_name, print_info=True)
            print('-'*100)
            print(f"Configuration update for endpoint {pt_endpoint_name} is complete.")
            
        else:
            print(f"Configuration update for endpoint {pt_endpoint_name} is still in progress or Check time-out reached.")

        break
    except HTTPError as e:
        if 'RESOURCE_CONFLICT' in str(e):
            print(f"Conflict error: {e}. Retrying in 30 seconds...")
            time.sleep(30)  # wait before retrying | e.g. to get endpoint status update from API
        else:
            raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2.4 Review Serving Endpoint UI for serving status and updated `ai_gateway` configs
# MAGIC
# MAGIC The update to provisioning process takes a while so you will see the endpoint with an `active` and `pending` configuration update status and the endpoint will also show that it is `(updating)` ... on the UI    
# MAGIC
# MAGIC ![]()
# MAGIC
# MAGIC ![](./imgs/05_systemai_endpt_pt_ABtest_pending.png)
# MAGIC
# MAGIC Once the configuration updates are ready the `pending` status will no longer show and the endpoint and served_entities will be in `ready` state. 
# MAGIC
# MAGIC ![]()
# MAGIC
# MAGIC ![](./imgs/05_systemai_endpt_pt_ABtest_ready.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 [Optional] Upate Existing `External` Endpoint Configs & AI Gateway Settings 
# MAGIC
# MAGIC Similarly, (if it was previously created) we can also update our `External` `Model Serving Endpoint` to serve 2 different external models (instead of just one model) and route its query traffic 50/50% to the served entities. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.3.1 By default, `update_external_endpoint` == `False` 
# MAGIC
# MAGIC As before you can refer to the code if you don't have the required API tokens for creating the external endpoint at this moment.

# COMMAND ----------

# DBTITLE 1,Define update_external_endpoint()
## Setting this to False as default: external endpoint will not be updated. 
# Change to True to update external endpoint already exists

update_external_endpoint = False

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.3.2 Specify New `External` Endpoint Configs & Updated AI Gateway Settings 

# COMMAND ----------

# DBTITLE 1,Compare | A/BTest 2 External Models
if update_external_endpoint:

    ## AzureOpenai API key in Databricks Secrets
    # SECRETS_SCOPE = "<secrets_scope>"
    # SECRETS_KEY = "AzureOPENAI_API_KEY" # key for AzureOpenAI_API_Token
    # Azure_RESOURCE_NAME | "openai_api_base": "https://<Azure_RESOURCE_NAME>.openai.azure.com/" 

    external_new_config = {
                            "served_entities": [
                                                ## MODEL A
                                                {
                                                "name": "gpt-4o-mini",
                                                "external_model": {
                                                    "name": "gpt-4o-mini",
                                                    "provider": "openai",
                                                    "task": "llm/v1/chat",
                                                    "openai_config": {
                                                            # "openai_api_key": f"{{{{secrets/{SECRETS_SCOPE}/{azopenai_SECRETS_KEY}}}}}",
                                                            # "openai_api_base": "https://<Azure_RESOURCE_NAME>.openai.azure.com/",
                                                            # "openai_deployment_name": "<your_deployment_name>",
                                                            # "openai_api_version": "<deployed_model_api_version>",
                                                            # "openai_api_type": "azure",                        
                                                        },
                                                    },
                                                },
                                                ## MODEL B
                                                {
                                                    "name": "gpt-4o",
                                                    "external_model": {
                                                        "name": "gpt-4o",
                                                        "provider": "openai",
                                                        "task": "llm/v1/chat",
                                                        "openai_config": {
                                                                # "openai_api_key": f"{{{{secrets/{SECRETS_SCOPE}/{azopenai_SECRETS_KEY}}}}}",
                                                                # "openai_api_base": "https://<Azure_RESOURCE_NAME>.openai.azure.com/",
                                                                # "openai_deployment_name": "<your_deployment_name>",
                                                                # "openai_api_version": "<deployed_model_api_version>",
                                                                # "openai_api_type": "azure",                        
                                                            },
                                                        },
                                                    },
                                                ],
                            
                            "traffic_config": {
                                "routes": [
                                    {"served_model_name": "gpt-4o-mini", "traffic_percentage": 50},
                                    {"served_model_name": "gpt-4o", "traffic_percentage": 50},
                                ],    
                            },

                            'tags': [{'key': 'owner', 'value': user_info},
                                    {'key': 'removeAfter', 'value': '2025-03-31'},
                                    {'key': 'tko-demo', 'value': 'true'} 
                                    ],                        
                        }

    aigateway_config_ext = {'usage_tracking_config': {'enabled': True},
                            # ## omit if already enabled 
                            # 'inference_table_config': {'catalog_name': f'{CATALOG_NAME}',
                            #                            'schema_name': f'{SCHEMA_NAME}',
                            #                            'table_name_prefix': f'{external_endpoint_name}',
                            #                            'enabled': True
                            #                           },
                            # ## remove limits for this example 
                            # 'rate_limits': [{'calls': 5, 'key': 'endpoint', 'renewal_period': 'minute'},
                            #                 {'calls': 5, 'key': 'user', 'renewal_period': 'minute'}
                            #                ],
                            
                            'guardrails': {'input': {'safety': False,
                                                    'pii_detection': True,
                                                    'pii': {'behavior': 'BLOCK'},
                                                    'invalid_keywords': ['SuperSecretProject', 'pw', 'password']
                                                    },
                                            'output': {'safety': False,
                                                    'pii_detection': True,
                                                    'pii': {'behavior': 'BLOCK'}
                                                    }
                                        }
                            }

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.3.3 Update `External` Endpoint Configs & AI Gateway Settings with `MLflowDeploymentManager()`

# COMMAND ----------

import time
from requests.exceptions import HTTPError

if update_external_endpoint:

    for _ in range(5):
        try:        
            ep_manager.update_endpoint_config(external_endpoint_name, external_new_config) # 1st update the endpoint config
            ep_manager.update_endpoint_ai_gateway(external_endpoint_name, aigateway_config_ext) # 2nd update the endpoint ai gateway config

            print('-'*100)       
            # ep_manager.get_endpoint(external_endpoint_name, print_info=True)

            if is_update_in_progress(ep_manager, external_endpoint_name, timeout=300):
                print('-'*100)
                ep_manager.get_endpoint(external_endpoint_name, print_info=True)
                print('-'*100)
                print(f"Configuration update for endpoint {external_endpoint_name} is complete.")
                
            else:
                print(f"Configuration update for endpoint {external_endpoint_name} is still in progress or Check time-out reached.")

            break
        except HTTPError as e:
            if 'RESOURCE_CONFLICT' in str(e):
                print(f"Conflict error: {e}. Retrying in 30 seconds...")
                time.sleep(30)  # wait before retrying | e.g. to get endpoint status update from API
            else:
                raise e

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 1.3.4 Review Serving Endpoint UI for serving status and updated `ai_gateway` configs

# COMMAND ----------

# MAGIC %md
# MAGIC Updating the external endpoint configs with the additional externally served entitity will also show up on the endpoint UI when `ready` -- this is relatively quick. 
# MAGIC
# MAGIC <!-- <img src="./imgs/05_external_endpt_pt_ABtest.png" alt="External Endpoint AB Test"> -->
# MAGIC
# MAGIC ![](./imgs/05_external_endpt_pt_ABtest.png)
# MAGIC
# MAGIC <!-- ![](/Workspace/Users/may.merkletan@databricks.com/REPOs/db_ai_gateway_lab/devs/imgs/05_external_endpt_pt_ABtest.png) -->

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Send queries to updated endpoint(s) using `ai_query()`
# MAGIC
# MAGIC With the configurations of our endpoint(s) updated, we can now send some synthetic query requests to test out the `query traffic routing` / `load-balancing` on the endpoint(s). 
# MAGIC
# MAGIC We will first generate some bulk queries and then send them via `ai_query()`. When the endpoint is created with provisioned throughput, these bulk queries can be sent as [Batch Inference](https://www.databricks.com/blog/introducing-simple-fast-and-scalable-batch-llm-inference-mosaic-ai-model-serving). 
# MAGIC
# MAGIC We will wait for the `Inference payload` to update and populate with these `requests` and corresponding `responses` before we take a look at how the queries have been routed. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Generate Some Synthetic Random Queries 
# MAGIC
# MAGIC We will first generate some  queries with the help of a query template. 

# COMMAND ----------

# DBTITLE 1,Generate Queries with non sensitive info. or PII
# generate_and_format_queries(query_templates, topics, sensitive_words, messageFlag, num_queries, percentage2inject)

query_templates = [
    "What is {topic}?",
    "How does {topic} work?",
    "Explain the concept of {topic}.",
    "Compare {topic1} and {topic2}.",
    "What are the benefits of {topic}?",
    "Describe the history of {topic}.",
    "What are the latest trends for {topic}?"
]

topics = ["artificial intelligence", "quantum computing", "blockchain", "machine learning", "cybersecurity", "data science", "travel", "sci-fi", "a protein", "being kind", "being a foodie"]

sensitive_words = ['']

messageFlag = "XXXXXXX AB_TEST XXXXXXX Let's start ai_query() requests!!! ---------------------"

num_queries = 50
percentage2inject = 0  # 0% chance to inject sensitive words

queries_df = generate_and_format_queries(query_templates, topics, sensitive_words, messageFlag, num_queries, percentage2inject)

display(queries_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2.2 Perform Batch Inference with `ai_query()` 
# MAGIC
# MAGIC When the endpoint is created with [provisioned throughput](https://docs.databricks.com/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis.html), these bulk queries can be sent as [Batch Inference](https://www.databricks.com/blog/introducing-simple-fast-and-scalable-batch-llm-inference-mosaic-ai-model-serving). 

# COMMAND ----------

# DBTITLE 1,ai_query() common parameters
# Define the prompt template
prompt_template = (
    'As a helpful and knowledgeable assistant, you will provide short and very succinct responses. '
    'Return the response as aiq_response key and corresponding value as strings in a {{"aiq_response":"<string>"}} json. '
    'Do not include text before and after the json. >> '
)

max_tokens = 50  
temperature = 0.7

# COMMAND ----------

# MAGIC %md 
# MAGIC Due to the [lazy nature of spark evaluation](https://medium.com/@john_tringham/spark-concepts-simplified-lazy-evaluation-d398891e0568) to `execute` the query command, we require an `"action"` (e.g. `display` or `count`) to `execute` the query command.  

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.1 Batch Inference *with* provisioned throughput systemai_endpt_pt_{} 
# MAGIC
# MAGIC **NB:** _Make sure that the provisioned throughput endpoint is 'warmed' up before sending `ai_query()` to leverage the more efficient batch inferencing._

# COMMAND ----------

# DBTITLE 1,systemai_endpt_pt_{}
from pyspark.sql import functions as F
import time

# Create TempView
queries_df.createOrReplaceTempView("queries_df")

# Define the endpoint and parameters
endpoint_name = pt_endpoint_name

# Start timing
start_time = time.time()

# Perform the ai_query and add the result as a new column
spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW result_df AS
    SELECT
      query,
      regexp_replace(
        ai_query(
          '{endpoint_name}',
          to_json(named_struct(
            'userAIQ', array(named_struct('prompt', CONCAT(
              '{prompt_template}',
              query
            ))),
            'max_tokens', {max_tokens},
            'temperature', {temperature}
          ))
        ),
        '\\\\+', '' -- regexp_replace to remove any number of backslashes in output request json
      ) AS response
    FROM queries_df
""")

# Display the result
display(spark.table("result_df"))

# End timing
end_time = time.time()

# Print execution time
execution_time = end_time - start_time
print(f"Execution time for {endpoint_name}: {execution_time} seconds")

# (** When PTendpt warmed up to leverage Batch Inference **) 
# Execution time for systemai_endpt_pt_mmt: 3.3393921852111816 seconds
# Execution time for systemai_endpt_pt_mmt: 3.1632046699523926 seconds

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.2 [Optional] Batch Inference with *without* provisioned throughput external_endpt_{} 
# MAGIC
# MAGIC **If you have already created an `external` endpoint,** you can also send the bulk queries to it.   
# MAGIC
# MAGIC If not, not to worry, you can use the following as example code for reference without running the code. 
# MAGIC
# MAGIC _**What we wanted to demonstrate here with is the difference between provisioned and non-provisioned throughput endpoints and how the provisioned throughput endpoint allows for more efficient [Batch Inference](https://docs.databricks.com/aws/en/machine-learning/model-inference/).**_

# COMMAND ----------

# DBTITLE 1,external_endpt_{}
from pyspark.sql import functions as F
import time

if update_external_endpoint:
    
  # Create TempView
  queries_df.createOrReplaceTempView("queries_df")

  # Define the endpoint and parameters
  endpoint_name = external_endpoint_name

  # Start timing
  start_time = time.time()

  # Perform the ai_query and add the result as a new column
  spark.sql(f"""
      CREATE OR REPLACE TEMP VIEW result_df AS
      SELECT
        query,
        regexp_replace(
          ai_query(
            '{endpoint_name}',
            to_json(named_struct(
              'userAIQ', array(named_struct('prompt', CONCAT(
                '{prompt_template}',
                query
              ))),
              'max_tokens', {max_tokens},
              'temperature', {temperature}
            ))
          ),
          '\\\\+', '' -- regexp_replace to remove any number of backslashes in output request json
        ) AS response
      FROM queries_df
  """)

  # Display the result
  display(spark.table("result_df"))

  # End timing
  end_time = time.time()

  # Print execution time
  execution_time = end_time - start_time
  print(f"Execution time for {endpoint_name}: {execution_time} seconds")

# Execution time for external_endpt_mmt: 73.47566866874695 seconds
# Execution time for external_endpt_mmt: 86.12629246711731 seconds

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.3 NOTEWORTHY
# MAGIC We have sent the same set of queries to both endpoints.   
# MAGIC
# MAGIC As a basic comparison of the `ai_query()` execution times between the 2 endpoints, you will notice that (_**when not starting from zero and warmed up**_) the **provisioned throughput endpoint serving `system.ai` Foundational Models was able to leverage [Batch Inferencing](https://docs.databricks.com/aws/en/machine-learning/model-inference/)** and completed the set of requests (in ~ 3--4s) roughly ~20x faster than the non-provisional throughput endpoint serving `external` Foundational Models (which took ~ 75--85s)! 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Assess Endpoint Traffic Routing
# MAGIC
# MAGIC We will wait for the `Inference payload` to update and populate with these `requests` and corresponding `responses` before we take a look at how the queries have been routed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Inference `payload` `requests` and `responses`
# MAGIC
# MAGIC Given the `served_entities_id`'s 'entity_name' or model name mapping information is found in different tables for `system.ai` or `external` serving endpoints' payload, we will **standardize the process with a table join between the `Inference payload` and `system.serving.served_entities` tables to extract the served `entity_name`** for checking our the traffic of our queries have been routed. We define  helper-functions here to assist with the task. 

# COMMAND ----------

# DBTITLE 1,Functions to load and join payload and served_entities tables
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def process_endpoint_data(endpoint_name):
    # Load the table into a DataFrame
    df = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.{endpoint_name}_payload")

    # Load the served_entities table
    served_entities_df = spark.table("system.serving.served_entities")

    # Join the df DataFrame with the served_entities_df to get served_entity_id
    df_joined = (df.join(
                  served_entities_df,
                  on = "served_entity_id",
                  how = "left")
                 .withColumn('model', F.col('entity_name'))
                 .drop(served_entities_df["served_entity_id"])
                 .select(
                        'request_date',
                        'databricks_request_id',
                        'client_request_id',
                        'status_code',
                        'request_time',    
                        df.served_entity_id.alias('served_entity_id'),
                        served_entities_df.entity_name.alias('served_entity_name'),
                        'sampling_fraction',
                        'execution_duration_ms',        
                        'request',
                        'response',      
                        'logging_error_codes',
                        'requester'
                       )
                 )

    return df_joined

def check_query_routing(df_joined, message):
    # Filter for the specific message
    df_filtered_message = df_joined.filter(
        F.col("request").contains(message)
    )

    # Find the latest occurrence of the message
    latest_timestamp = df_filtered_message.agg(F.max("request_time")).collect()[0][0]

    # Define a window specification
    window_spec = Window.partitionBy()

    
    # Filter and count subsequent queries sent to different entity_name or model
    df_queries_after_latest = (df_joined
                               .filter(F.col("status_code") == 200)
                               .filter(F.col("request_time") > latest_timestamp)                            
                               .groupBy("served_entity_name", "status_code").count()
                               .withColumn('percentage', 
                                           F.round(F.col('count') / F.sum('count').over(window_spec),2))
                               )

    # Display the result
    display(df_queries_after_latest)


# Example usage

# df_joined = process_endpoint_data(pt_endpoint_name)
# messageFlag = "XXXXXXX Let's start ai_query() requests!!! ---------------------"
# check_query_routing(df_joined, messageFlag)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Assess Query Routing on Endpoints 
# MAGIC
# MAGIC Using  our `messageFlag` as `timestamp` `marker` to check on the frequency of requests sent to each of the `served_entity_ids`, we observe a rough `50/50` percentage split in the query traffic for both `system.ai` and `external` model serving endpoints, each serving 2 different model variants.  
# MAGIC
# MAGIC It is worth monitoring the traffic routing over time to ensure that expected load-balancing is achieved. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2.1 Query Routing on `systemai_endpt_pt_{}`

# COMMAND ----------

# DBTITLE 1,systemai_endpt_pt_{}
df_joined = process_endpoint_data(pt_endpoint_name)
messageFlag = "XXXXXXX AB_TEST XXXXXXX Let's start ai_query() requests!!! ---------------------"
check_query_routing(df_joined, messageFlag)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2.2 Query Routing on `external_endpt_{}`

# COMMAND ----------

# DBTITLE 1,external_endpt_{}
df2_joined = process_endpoint_data(external_endpoint_name)
messageFlag = "XXXXXXX AB_TEST XXXXXXX Let's start ai_query() requests!!! ---------------------"
check_query_routing(df2_joined, messageFlag)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. NEXT
# MAGIC
# MAGIC Now that we have gone over how to set up serving **`system.ai`** and **`external`** **Foundational Models** as **`Endpoints`** as well as configuring the **[`AI Gateway`](https://docs.databricks.com/aws/en/ai-gateway)** settings, _**you are ready to try these out in your workspace.**_
# MAGIC
