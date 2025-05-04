# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 03 Enable Endpoint Rate Limits with AI Gateway
# MAGIC
# MAGIC As we continue to investigate the inference table payload, we will notice some steep spikes in usage suggesting a higher-than-expected volume of queries, e.g. bulk batch inferencing that we just sent to the endpoint.    
# MAGIC
# MAGIC ![02_03_BatchInference_EndpointMetrics.png](./imgs/02_03_BatchInference_EndpointMetrics.png).  
# MAGIC
# MAGIC
# MAGIC Extremely high usage could be costly if not monitored and limited.
# MAGIC
# MAGIC For illustrative purpose, we will simulate different users with the `requester` column to demonstrate different usage scenarios. In reality and depending on use case, the `requesters` will likely be different users, and their usage will vary depending on what they use the endpoint for e.g. development testing or application query etc. 
# MAGIC
# MAGIC In this notebook, we will walk through how to set up the **`AI Gateway`** configs to rate limit queries either by `endpoint` or `per user`. 

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
# MAGIC
# MAGIC #### `Databricks API TOKENS`
# MAGIC
# MAGIC In this notebook you will need a Databricks `API KEY TOKEN` to make calls to the `systemai` and/or `external` endpoint(s).    
# MAGIC If you have not already set up a workspace [Personal Access Token](https://docs.databricks.com/aws/en/dev-tools/auth/pat) or [Service Principal](https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m) and [register/manage the respective token secrets](https://docs.databricks.com/aws/en/security/secrets/) in [Databricks Secrets Utility](https://docs.databricks.com/aws/en/dev-tools/databricks-utils#secrets-utility-dbutilssecrets); please follow the relevant links to create them  before continuing. 

# COMMAND ----------

# DBTITLE 1,Databricks PAT/SP TOKEN
## Example workspace Personal Access Token or Service Principal API key in Databricks Secrets
DATABRICKS_SECRETS_SCOPE = "<workspace_secrets_scope>" # scope name associated to secret_key where its key_value is stored
DATABRICKS_SECRETS_KEY = "<databricks_{PAT/SP}token>" # key for workspace Personal Access Token (PAT) / Service Principal (SP)

DATABRICKS_API_TOKEN = dbutils.secrets.get(DATABRICKS_SECRETS_SCOPE, DATABRICKS_SECRETS_KEY) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Quick Check on Endpoint Usage 
# MAGIC
# MAGIC To begin we can take a look at the number of queries made per minute (or second or hour or day etc.) for ourselves as the user for the last 20 hours, groupby time-interval of `minute` and `requester` and counting the unique `databricks_request_id` as `queries_per_minute` (QPM). 
# MAGIC
# MAGIC ***IMPT NOTE***    
# MAGIC **_`For the visualizations we discuss here we had sent at least 2000 queries that were loggged in the corresponding payload.    
# MAGIC Given that your workspace environment and endpoint setup and responses may not be identical to the one used in developing these lab assets, the queries that produced the following plots we discuss here may not yield the SAME visualization output, i.e. the queries per minute `(QPM)` lines will likely differ, as will the inferences.    
# MAGIC We will use screenshots saved of these outputs for discussing the context of this notebook. Nonethelss, we are providing the example code and output (hidden + can be shown using `<0>` eye button or kebab cell menu) that was used to generate these plots if you wish to try them out for yourselves.`_**

# COMMAND ----------

# MAGIC %md
# MAGIC As illustrated in the visualisation(s) below, our query usage varies over time. 
# MAGIC ![03_user_QPM.png](./imgs/03_user_QPM.png)

# COMMAND ----------

# DBTITLE 1,User QPM (ref code + output viz hidden)
# from pyspark.sql.functions import date_trunc, col, countDistinct, current_timestamp
from pyspark.sql import functions as F, types as T

# Define the DataFrame
df = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.`{ENDPOINT_NAME}_payload`")

# Filter the DataFrame
filtered_df = df.filter(F.col("request_time") >= F.current_timestamp() - F.expr("INTERVAL 20 HOURS"))

# Group by and aggregate
result_df = (filtered_df
             .groupBy(F.date_trunc("minute", F.col("request_time")).alias("minute"),
                      "requester",
                      ) 
             .agg(F.countDistinct("databricks_request_id").alias("queries_per_minute")) 
             .orderBy(F.col("minute").desc())
            )

# Display the result
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC To keep the examples contained and avoid unnecessarily probing of unknown endpoints or other users' activities on the same workspace, we will use our payload data to simulate scenarios where certain users made high numbers of requests to our endpoint. 
# MAGIC
# MAGIC Users with permissions to access `system.serving.endpoint_usage` table, can monitor queries per {`time_inveval of choice`} by users for `workspace_id`, `account_id`, or `served_entiy_id` can be investigated for `requester` as well as `input` and `output` `token` or `character` `counts`, `inference execution duration`, and/or whether the `request` response is in a `streamed` mode.   

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scenario #1
# MAGIC In the following set of visualization we see 3 users querying the same endpoint at roughly the same time with indivually different frequency. For example `user2` consistently makes more queries than other users. 
# MAGIC ![](./imgs/03_MultiUsers_QPM_vis2a.png) 
# MAGIC <!-- ![](./imgs/03_MultiUsers_QPM_vis2b.png) -->
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Muti-user QPM Scenario#1 (ref code + output viz hidden)
from pyspark.sql import functions as F

# Define the DataFrame
df = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.`{ENDPOINT_NAME}_payload`")

# Add a random column with a seed
seed = 42
df_with_random = df.withColumn("random", F.rand(seed))

# Replace 5% of the requester values with two different user names
df_replaced = df_with_random.withColumn(
    "requester",
    F.when(F.col("random") < 0.125, "user1@example.com")
     .when(F.col("random") < 0.655, "user2@example.com")
     .otherwise(F.col("requester"))
).drop("random")

# Filter the DataFrame
filtered_df = df_replaced.filter(F.col("request_time") >= F.current_timestamp() - F.expr("INTERVAL 20 HOURS"))

# Group by and aggregate
result_df = (filtered_df
             .groupBy(F.date_trunc("minute", F.col("request_time")).alias("minute"),
                      "requester",
                      ) 
             .agg(F.countDistinct("databricks_request_id").alias("queries_per_minute")) 
             .orderBy(F.col("minute").desc())
            )

# Display the result
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scenario #2
# MAGIC In another scenario shown in the following set of visualization we see 3 users querying the same endpoint with relative higher frequency distribution in a time period compared to other users. For example `user2` has hiqh query usage at the beginning of the 20 hours x-axis range, while `user1` is crushing the endpoint query towards the end of the 20 hour period. 
# MAGIC
# MAGIC ![](./imgs/03_MultiUsers_QPM_vis3a.png)
# MAGIC <!-- ![](./imgs/03_MultiUsers_QPM_vis3b.png) -->

# COMMAND ----------

# DBTITLE 1,Multi-User QPM Scenario#2 (ref code + output viz hidden)
from pyspark.sql import functions as F

# Define the DataFrame
df = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.`{ENDPOINT_NAME}_payload`")

# Add a random column with a seed
seed = 42
df_with_random = df.withColumn("random", F.rand(seed))

# Replace 5% of the requester values with two different user names
df_replaced = df_with_random.withColumn(
    "requester",
    F.when(F.col("random") < 0.025, "user1@example.com")
     .when(F.col("random") < 0.245, "user2@example.com")
     .otherwise(F.col("requester"))
).drop("random")

# Filter the DataFrame
filtered_df = df_replaced.filter(F.col("request_time") >= F.current_timestamp() - F.expr("INTERVAL 20 HOURS"))

# Simulate variable requester usage for each time interval
df_variable_usage = filtered_df.withColumn(
    "requester",
    F.when(F.hour(F.col("request_time")) % 3 == 0, "user1@example.com")
     .when(F.hour(F.col("request_time")) % 3 == 1, "user2@example.com")
     .otherwise(F.col("requester"))
)

# Group by and aggregate
result_df = (df_variable_usage
             .groupBy(F.date_trunc("minute", F.col("request_time")).alias("minute"),
                      "requester",
                      ) 
             .agg(F.countDistinct("databricks_request_id").alias("queries_per_minute")) 
             .orderBy(F.col("minute").desc())
            )

# Display the result
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ---    
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure Endpoint AI Gateway Rate Limits 
# MAGIC
# MAGIC There will likely be different combination of query usage patterns for each endpoint.
# MAGIC
# MAGIC To keep Endpoint Query Rates consistent, especially if it is important to maintain a steady cost billing, we can set a rate limit to prevent excessive queries. 
# MAGIC
# MAGIC We will review how **`AI Gateway Endpoint Configurations`** allow you to set query limit on the `endpoint`, as well as the option to set `per-user` query limits.
# MAGIC
# MAGIC For the purpose of illustration we will set the Query_per_minute Rate to be very low: e.g. `5 QPS` so that we can exceed this limit when testing. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Endpoint `ai_gateway` Config. Update Process: [`Rate Limits`]

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.1 Leverage our pre-defined `MLflowDeploymentManager()` to execute the endpoint cofiguration updates

# COMMAND ----------

# DBTITLE 1,MLflowDeploymentManager
# utils.py -- has the class MLflowDeploymentManager defined 

# Instantiate the manager
ep_manager = MLflowDeploymentManager()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.2 Exisiting Endpoint `ai_gateway` Config: `pt_ep_config_aigateway`
# MAGIC Let's check the `ai_gateway` configurations of our provisioned throughput endpoint `systemai_endpt_pt_{}`:   
# MAGIC
# MAGIC We have so far enabled `usage_tracking_config` and `inference_table_config`.    
# MAGIC
# MAGIC We notice that `guardrails` have not yet been set. We will walk through the configuration of these in the following notebook.    
# MAGIC
# MAGIC **Currently, there is also no sign of any `rate_limits` settings, which we will go ahead and add them as the focus of this notebook.** 

# COMMAND ----------

# DBTITLE 1,Existing endpoint AI Gateway Configs
## we can use the client to get the existing AI Gateway Configs 
ep_info = ep_manager.get_endpoint(ENDPOINT_NAME)
pt_ep_config_aigateway = ep_info['ai_gateway']  

pt_ep_config_aigateway

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.1.3 Specify Endpoint `ai_gateway` settings for desired `rate_limits`
# MAGIC
# MAGIC We will first update the endpoint `ai_gateway` config dictionary with the desired `rate_limits`. 
# MAGIC
# MAGIC You have the option to limit endpoint query rates for the `endpoint` and/or `user` and specify the `renewal_period`. 
# MAGIC
# MAGIC Here we will limit the `query-per-minute` rate for both the `endpoint` and `user`.

# COMMAND ----------

# DBTITLE 1,Update AI Gateway Rate Limit Endpoint Configs
## update our AI Gateway Configs settings with rate_limits
pt_ep_config_aigateway.update({"rate_limits": [
                                                {"calls": 5, "key": "endpoint", "renewal_period": "minute"},
                                                {"calls": 5, "key": "user", "renewal_period": "minute"}
                                               ]
                               }
                              )

# check
pt_ep_config_aigateway                              

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.4 Push `ai_gateway` configuration with `rate_limits` updates to endpoint
# MAGIC
# MAGIC With the updated `ai_gateway` configurations with `rate_limits` defined, we can use our instantiated endpoint manager client (`ep_manager`) to push the updated `ai_gateway` configuration settings on the endpoint 

# COMMAND ----------

# DBTITLE 1,Client Update Endpoint AI Gateway Configs
# Use MLflowDeploymentManager to update AI Gateway Configs 
ep_manager.update_endpoint_ai_gateway(pt_endpoint_name, pt_ep_config_aigateway)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.5 Review the updated `ai_gateway` configuration for `rate_limits`
# MAGIC
# MAGIC We can call a `get_endpoint` info. to review the updated configuration settings and verify that the update was successful. The `rate_limits` settings we specified should show up within the `ai_gateway` key-value in the configuration dictionary.  

# COMMAND ----------

# DBTITLE 1,Check Endpoint Configs
ep_manager.get_endpoint(pt_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.6 Review Serving Endpoint UI for updated `ai_gateway` configs 
# MAGIC
# MAGIC If we next check on the Serving endpoints UI page, we should also notice that the `AI Gateway` configuration for `Rate limits` now shows the QPM values for `(per user)` and `(per endpoint)`:

# COMMAND ----------

# MAGIC %md
# MAGIC ![03_RateLimits_systemai.png](./imgs/03_RateLimits_systemai.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 [Optional] External Endpoint `ai_gateway` Config. Update
# MAGIC
# MAGIC If you have previously created it with the relevant pre-requisites satisfied, we can also do the same for the external endpoint `external_endpt_{endpt_name_suffix}`:  

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.1 By default: `update_external_endpoint` variable is set to  `False` 
# MAGIC As before you can refer to the code if you don't have the required API tokens for creating the external endpoint at this moment.

# COMMAND ----------

# DBTITLE 1,Define update_external_endpoint
## Setting this to False as default: external endpoint will not be updated. 
# Change to True to update external endpoint already exists
update_external_endpoint = False

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.2 Update External Endpoint `ai_gateway` Config: `ext_ep_config_aigateway`
# MAGIC
# MAGIC If `update_external_endpoint == True` the following code snippet will first extract exisitng `ai_gateway` config for the external endpoint `external_endpt_{}` as a reference. 
# MAGIC
# MAGIC Since it is our first time updating the external endpoint's `ai_gateway` configuration, we will next specify the configurations for `usage_tracking_config`, `inference_table_config`, as well as `rate_limits`.  
# MAGIC
# MAGIC Subsequently we will call the function `update_endpoint_ai_gateway` to push these updated settings to the endpoint using our instantiated endpoint manager client `ep_manager` 

# COMMAND ----------

# DBTITLE 1,external_endpt_{}
if update_external_endpoint: 

    external_endpoint_name = f"external_endpt_{endpt_name_suffix}"

    ## Existing ai_gateway config
    ext_ep_info = ep_manager.get_endpoint(external_endpoint_name)
    ext_ep_config_aigateway = ext_ep_info['ai_gateway']  
    print(json.dumps(ext_ep_config_aigateway, indent=4))

    ## Since this is the first time we are configuring the ai-gateway for our external endpoint, we specify the various parameters in the dictionary.
    ext_ep_config_aigateway = {
                                ## Enable usage tracking to track the number of requests
                                "usage_tracking_config": {
                                "enabled": True
                                },
                                ## Enable payload logging to log the request and response
                                "inference_table_config": {
                                    "enabled": True,
                                    "catalog_name": CATALOG_NAME,
                                    "schema_name": SCHEMA_NAME
                                },
                                "rate_limits": [
                                    {"calls": 5, "key": "endpoint", "renewal_period": "minute"},
                                    {"calls": 5, "key": "user", "renewal_period": "minute"}
                                ]
                            }
    print("-"*100)
    print(json.dumps(ext_ep_config_aigateway, indent=4))

    ext_ep_config_aigateway_update = ep_manager.update_endpoint_ai_gateway(external_endpoint_name, ext_ep_config_aigateway)
    print(json.dumps(ext_ep_config_aigateway_update, indent=4))

    ep_manager.get_endpoint(external_endpoint_name, print_info=True)    

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.2.3 Review Updates in Serving Endpoint UI 
# MAGIC
# MAGIC If you previously created an existing `external` endpoint and updated its `ai_gateway` as described above e.g. to include `Rate Limits`, it should look similar to the Serving endpoints UI screenshot here: 

# COMMAND ----------

# MAGIC %md
# MAGIC ![03_RateLimits_external.png](./imgs/03_RateLimits_external.png)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 3. TEST Enabled AI Gateway Rate Limits
# MAGIC
# MAGIC To check that the updated **`AI Gateway Configs`** are in place for **`Rate Limits`**, we can test sending more queries per minute than the `5 QPM` Limit. 
# MAGIC
# MAGIC We will use the [Databricks REST API](https://docs.databricks.com/api/workspace/introduction) to test which would typically require a user [Personal Access Tokens (PAT)](https://docs.databricks.com/aws/en/dev-tools/auth/pat) &/or [Service Principal (SP)](https://docs.databricks.com/aws/en/dev-tools/auth#service-principal). 
# MAGIC
# MAGIC Here for the Lab, we will **use [Databricks Utilitites `dbutils()` functionality](https://docs.databricks.com/aws/en/dev-tools/databricks-utils)** to help us extract **`an internal token provided by Databricks for the current notebook context`**, as the required **`api_key`** in this example. This internal token is suitable for making API requests within the same Databricks workspace.    
# MAGIC
# MAGIC (NB: If you need to authenticate API requests from external tools or scripts, you should use a PAT/SP or an OAuth token instead.) 

# COMMAND ----------

# DBTITLE 1,Functions to run API test
import time
import requests
import json

messageFlag = '>>>>>>>>>> TESTING UPDATES TO RATE LIMITS !!! <<<<<<<<<<<'

# Function to set up parameters
def setup_params(endpoint_name):
    # Get the workspace URL from the Databricks notebook context
    DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
    # print(f"DATABRICKS_HOST: {DATABRICKS_HOST}")

    # Define the endpoint_url
    endpoint_url = f"{DATABRICKS_HOST}/serving-endpoints/{endpoint_name}/invocations"

    # Get the API key from the Databricks notebook context | You can also use PAT (Personal Access Token) or Service Principal Token for the API key required to access the REST API.
    # If not already defined above
    # DATABRICKS_API_TOKEN = dbutils.secrets.get(DATABRICKS_SECRETS_SCOPE, DATABRICKS_SECRETS_KEY) 
    

    # Define the headers
    headers = {
        "Authorization": f"Bearer {DATABRICKS_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    return endpoint_url, headers


# Function to format the request to be sent
def send_request(i, endpoint_url, headers, messageFlag):
    if i == 0:
        # message_content = '>>>>>>>>>> TESTING UPDATES TO RATE LIMITS !!! <<<<<<<<<<<'
        message_content = messageFlag
    else:
        message_content = f"This is request {i}"
    
    data = {
        "messages": [
            {"role": "user", "content": message_content}
        ],
        "max_tokens": 10
    }
    response = requests.post(endpoint_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()


# Function to send multiple requests
def send_Nrequests(endpoint_name, messageFlag):
    endpoint_url, headers = setup_params(endpoint_name)
    start_time = time.time()
    for m in range(20):
        try:
            response = send_request(m, endpoint_url, headers, messageFlag)
            print(f"Request {m} sent, Response: {response['choices'][0]['message']['content']}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit exceeded
                print(e)
                print(f"Endpoint: {endpoint_url.split('/')[-2]} -- Rate Limit exceeded at request {m}. Waiting for 60 seconds.")
                time.sleep(60)  # Wait for 60 seconds before retrying
            else:
                print(f"Error occurred at request {m}: {e}")
                break
        except Exception as e:
            print(f"Error occurred at request {m}: {e}")
            break
    print(f"Total time: {time.time() - start_time:.2f} seconds")


# Example Usage
# send_Nrequests(endpoint_name, messageFlag)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## check & TEST wrt api tokens
# MAGIC

# COMMAND ----------

# DBTITLE 1,Functions to run API test
import time
import requests
import json

messageFlag = '>>>>>>>>>> TESTING UPDATES TO RATE LIMITS !!! <<<<<<<<<<<'

# Function to set up parameters
def setup_params(endpoint_name):
    # Get the workspace URL from the Databricks notebook context
    DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
    # print(f"DATABRICKS_HOST: {DATABRICKS_HOST}")

    # Define the endpoint_url
    endpoint_url = f"{DATABRICKS_HOST}/serving-endpoints/{endpoint_name}/invocations"

    # We will use PAT (Personal Access Token) or Service Principal Token (derived from STORED SECRETS above) for the API key required to access the REST API.
    # If not already defined above
    # DATABRICKS_API_TOKEN = dbutils.secrets.get(DATABRICKS_SECRETS_SCOPE, DATABRICKS_SECRETS_KEY) 
    
    # Define the headers
    headers = {
        "Authorization": f"Bearer {DATABRICKS_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    return endpoint_url, headers


# Function to format the request to be sent
def send_request(i, endpoint_url, headers, messageFlag):
    if i == 0:
        # message_content = '>>>>>>>>>> TESTING UPDATES TO RATE LIMITS !!! <<<<<<<<<<<'
        message_content = messageFlag
    else:
        message_content = f"This is request {i}"
    
    data = {
        "messages": [
            {"role": "user", "content": message_content}
        ],
        "max_tokens": 10
    }
    response = requests.post(endpoint_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()


# Function to send multiple requests
def send_Nrequests(endpoint_name, messageFlag):
    endpoint_url, headers = setup_params(endpoint_name)
    start_time = time.time()
    for m in range(20):
        try:
            response = send_request(m, endpoint_url, headers, messageFlag)
            print(f"Request {m} sent, Response: {response['choices'][0]['message']['content']}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit exceeded
                print(e)
                print(f"Endpoint: {endpoint_url.split('/')[-2]} -- Rate Limit exceeded at request {m}. Waiting for 60 seconds.")
                time.sleep(60)  # Wait for 60 seconds before retrying
            else:
                print(f"Error occurred at request {m}: {e}")
                break
        except Exception as e:
            print(f"Error occurred at request {m}: {e}")
            break
    print(f"Total time: {time.time() - start_time:.2f} seconds")


# Example Usage
# send_Nrequests(endpoint_name, messageFlag)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Rate Limits Test: `systemai_endpt_pt_{} url`
# MAGIC
# MAGIC Let's test to see if the `rate_limits` enabled on our provisioned throughput endpoint serving `system.ai.llama_v3_3_70b_instruct` Foundational Model endpoint is working as intended: 

# COMMAND ----------

# DBTITLE 1,systemai_endpt_pt_{} url -- API test
send_Nrequests(pt_endpoint_name, messageFlag)

## If this appears to take a while, interrupt and rerun the cell.
## It is likely due to the provisioned throughput endpoint having scaled to zero and needs time to warm-up again.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Rate Limits Test: `external_endpt_{} url`
# MAGIC
# MAGIC Similarly we can also test the `external_endpt` serving Azure OpenAI `gpt-4o-mini` model. 

# COMMAND ----------

# DBTITLE 1,external_endpt_{} url -- API test
if update_external_endpoint: 
  send_Nrequests(pt_endpoint_name, messageFlag)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 QPM > set threshold : `429 status code` triggered
# MAGIC
# MAGIC As we see, the updated `rate_limits` for our `ai_gateway` configs. kicked in and the `429 status code` was triggered when [too many requests per minute are being sent](https://github.com/mlflow/mlflow/blob/master/mlflow/gateway/constants.py#L22); other retry codes include (`500: Server Error`, `502: Bad Gateway`, `503: Service Unavailable`). 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 3.3.1 Check `pt_endpoint_name` `payload` for `429 status code`
# MAGIC
# MAGIC When the corresponding `request` and `response` in `payload` table are updated (~ 30+mins later), we can check IF the `429 status_code` is being logged as well.    
# MAGIC
# MAGIC We will use the _**`helper_function`**_ **`parse_payload()`** to help here:

# COMMAND ----------

# DBTITLE 1,payload for systemai_pt_endpt_{}
from pyspark.sql import functions as F, types as T

payload_tablename = f"{CATALOG_NAME}.{SCHEMA_NAME}.{ENDPOINT_NAME}_payload"
# request_date = F.current_date() #default for function
# messageFlag = '>>>>>>>>>> TESTING UPDATES TO RATE LIMITS !!! <<<<<<<<<<<'
filter_string = messageFlag 


pt_ep_payload_df = parse_payload(payload_tablename, filter_string=filter_string)

display(pt_ep_payload_df.select("request_time", "databricks_request_id",  
                          "status_code", "request_messages_user_query", "response_messages", 
                          "prompt_tokens", "completion_tokens", "total_tokens")
        .sort("request_time")
        )

## If the payload doesn't update, please wait a few minutes and try again. 

pt_ep_payload_df.filter(F.col('request_messages_user_query')!=messageFlag).groupBy("status_code").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3.2 Check `external_endpt_{}` `payload` for `429 status code` 
# MAGIC
# MAGIC Likewise, we can do similar and check the `external` endpoint `payload`, if it was created and its `ai_gateway` settings have been updated with the desired `rate_limits`.

# COMMAND ----------

# DBTITLE 1,payload for external_endpt_{}
from pyspark.sql import functions as F, types as T

if update_external_endpoint:

  ext_payload_tablename = f"{CATALOG_NAME}.{SCHEMA_NAME}.{EXT_ENDPOINT_NAME}_payload"
  # request_date = F.current_date() #default for function
  # messageFlag = '>>>>>>>>>> TESTING UPDATES TO RATE LIMITS !!! <<<<<<<<<<<' 
  filter_string = messageFlag 

  ext_ep_payload_df = parse_payload(ext_payload_tablename, filter_string=filter_string)

  display(ext_ep_payload_df.select("request_time", "databricks_request_id",  
                            "status_code", "request_messages_user_query", "response_messages", 
                            "prompt_tokens", "completion_tokens", "total_tokens")
          .sort("request_time")
          )
  
  ext_ep_payload_df.filter(F.col('request_messages_user_query')!=messageFlag).groupBy("status_code").count().show()

## If the payload doesn't update, please wait a few minutes and try again. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3.3 QPM Rate Limit `429 status code `doesn't get sent to the `payload` logging
# MAGIC
# MAGIC Checking the `payload` responses we observe that even though when too many requests are sent, the triggered `429 status code `doesn't get sent to the `payload` logging. Nonetheless, the interactively 
# MAGIC  printed `user_query` & `response` in our Rate Limits Tests (Section: 3.1, 3.2) earlier helps to show that certain queries e.g. `This is request #{e.g. 5,11,17}` do not show up due to the `429 status_code` being triggered when `Rate Limit` has been reached, as observed during testing above. 
# MAGIC
# MAGIC For `external` model endpoint, the `429 status code` gets triggered  when e.g. the Azure OpenAI pricing tier Token Rate Limits have been exceeded:  
# MAGIC
# MAGIC ```
# MAGIC {
# MAGIC     "error": "Received error from openai",
# MAGIC     "external_model_message": {
# MAGIC         "error": {
# MAGIC             "code": "429",
# MAGIC             "message": "Requests to the ChatCompletions_Create Operation under Azure OpenAI API version 2024-08-01-preview \n
# MAGIC             have exceeded token rate limit of your current OpenAI S0 pricing tier. Please retry after 38 seconds. \n
# MAGIC             Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit."
# MAGIC         }
# MAGIC     }
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Note on `429 status code` and Automatic Retry
# MAGIC
# MAGIC When the rate of requests exceeds the configured Rate Limit, a `429 Too Many Requests status code` will be returned, indicating that the user has sent too many requests in a given amount of time ("rate limiting"). This status is used by APIs to signal that the client should slow down the rate of requests.  
# MAGIC
# MAGIC Databricks [Mosaic AI Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving) endpoints implements an automatic retry. This feature ensures high availability and reliability for model serving. As such, we will not necessarily observe this `429 status code` being logged and instead, it may be experienced as a slight pause before new queries are processed.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. NEXT
# MAGIC We'll revisit our inference `payload` for sensitive info. and review how to configure **`Guardrails`** at our endpoint(s) with AI Gateway. 
