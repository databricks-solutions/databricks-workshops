# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Create and Serve Foundational Model Endpoint(s)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC [Mosaic AI Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html) supports real time and batch inference for [custom](https://docs.databricks.com/en/machine-learning/model-serving/custom-models.html) and [foundational](https://docs.databricks.com/en/machine-learning/model-serving/foundation-model-overview.html) models.
# MAGIC
# MAGIC There are different ways to  [create and serve foundational-model endpoints](https://docs.databricks.com/en/machine-learning/model-serving/create-foundation-model-endpoints.html#)    
# MAGIC e.g. via the [UI](https://docs.databricks.com/en/machine-learning/model-serving/create-foundation-model-endpoints.html#language-Serving%C2%A0UI), [`mlflow.deployments`](https://mlflow.org/docs/latest/python_api/mlflow.deployments.html?highlight=deployments#mlflow.deployments.DatabricksDeploymentClient), [serving-endpoints API](https://docs.databricks.com/api/workspace/servingendpoints/create), or [databricks sdk](https://databricks-sdk-py.readthedocs.io/en/latest/dbdataclasses/serving.html#databricks.sdk.service.serving.CreateServingEndpoint). 
# MAGIC
# MAGIC In this notebook we will learn how to programmatically create and serve `foundational` models using [`mlflow.deployments.DatabricksDeploymentClient()`](https://mlflow.org/docs/latest/python_api/mlflow.deployments.html?highlight=deployments#mlflow.deployments.DatabricksDeploymentClient).  
# MAGIC
# MAGIC ***NB: Given the pace of developments -- please check for configuration updates and version changes to*** [_`mlflow.deployments.DatabricksDeploymentClient()`_](https://mlflow.org/docs/latest/python_api/mlflow.deployments.html#mlflow.deployments.DatabricksDeploymentClient) & [workspace/servingendpoints API refs](https://docs.databricks.com/api/workspace/servingendpoints/create) ***and modify configuration requirements accordingly.***
# MAGIC
# MAGIC Functional Capabilities for _Client_ [as of period of lab development -- using `mlflow v 2.20.0`]:
# MAGIC - [`.create_endpoint()`](https://docs.databricks.com/api/workspace/servingendpoints/create)
# MAGIC - [`.update_endpoint_config()`](https://docs.databricks.com/api/workspace/servingendpoints/updateconfig)
# MAGIC - [`.update_endpoint_tags()`](https://docs.databricks.com/api/workspace/servingendpoints/patch) 
# MAGIC - [`.update_endpoint_ai_gateway()`](https://docs.databricks.com/api/workspace/servingendpoints/putaigateway)
# MAGIC - [`.delete_endpoint()`](https://docs.databricks.com/api/workspace/servingendpoints/delete) 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Run utils.py to access helper functions & set up configs
# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC #### `* Enter variables for your workspace *`
# MAGIC
# MAGIC Enter your catalog, schema and endpoint names for the variables in the cell below.

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
# MAGIC ## 1. Define Model Endpoint Configs.
# MAGIC
# MAGIC We will be creating a Foundational Model endpoint with [provisioned throughput](https://docs.databricks.com/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis.html) serving one of the `system.ai` models to use for the main Lab walkthrough. The 2nd [Optional] endpoint we will create will be serving an `External` Foundational Model using Azure OpenAI API. You can also create an  serving endpoint with other `external` Foundational Models in a similar fashion as long as some pre-requisites are met (and we will go over them).

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Define a Model Endpoint Configuration serving a `system.ai` model with Provisioned Throughput

# COMMAND ----------

# DBTITLE 1,system.ai endpoint config parameters
pt_endpoint_name = f"systemai_endpt_pt_{endpt_name_suffix}" 

pt_model_name = "system.ai.llama_v3_3_70b_instruct"
pt_model_version = 1

pt_ep_config = {                        
                "served_entities": [
                    {   
                        "name": pt_endpoint_name,

                        "entity_name": pt_model_name,
                        "entity_version": pt_model_version,
                        "min_provisioned_throughput": 0,
                        "max_provisioned_throughput": 9500,
                        "scale_to_zero_enabled": True,
                    }
                ],  
                ## Tagging is helpful for: Categorization and Labeling | Cost Monitoring | Lifecycle Management |  Access Control and Environment Management | Custom Identifiers
                "tags": [
                        {"key": "owner", "value": user_info},
                        {"key": "removeAfter", "value": "2025-03-31"},
                        {"key": "tko-demo", "value": "true"}
                        ]  

              }

# COMMAND ----------

pt_ep_config

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 [Optional] Define a Model Endpoint Configuration serving an `External` model
# MAGIC
# MAGIC There are various [external foundational models](https://docs.databricks.com/en/generative-ai/external-models/index.html) that one can choose to serve as an endpoint.  
# MAGIC
# MAGIC In this example walkthrough, we will use the Azure OpenAI `gpt4o-mini` to illustrate how this is achieved.
# MAGIC
# MAGIC ##### The following pre-requisites are needed: 
# MAGIC
# MAGIC - [Azure OpenAI API subscription](https://portal.azure.com/#create/Microsoft.CognitiveServicesOpenAI) (Keys & Endpoint Info.)
# MAGIC   - API Base = Endpoint URL: `https://<resourcename>.openai.azure.com/` 
# MAGIC   - API KEY: _to store as [`Databricks CLI`](https://docs.databricks.com/en/dev-tools/cli/index.html) [`secret` within a `scope`](https://docs.databricks.com/en/security/secrets/index.html)_
# MAGIC
# MAGIC - [Azure AI Foundry/Azure OpenAI Service/Deployments](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal#deploy-a-model) 
# MAGIC   - Deploy required external model
# MAGIC     - Deployment Name (under Deployment Info.): e.g. `gpt-4o-mini-2024-07-18` (NB model version = `2024-07-18`)
# MAGIC     - Model API Version (from Endpoint Target URI): e.g. `2024-08-01-preview`
# MAGIC
# MAGIC - Workspace [`Personal Access Token`](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/pat) OR [Azure Entra ID Token-based](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/service-prin-aad-token) [`Service Principal`](https://learn.microsoft.com/en-us/azure/databricks/admin/users-groups/service-principals) (recommended for production workloads) 
# MAGIC
# MAGIC ---    
# MAGIC
# MAGIC Ref: [`mlflow.deployments` -- configuring-the-gateway-server](https://mlflow.org/docs/latest/llms/deployments/index.html#configuring-the-gateway-server)
# MAGIC
# MAGIC  

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2.1 `create_external_endpoint = False`

# COMMAND ----------

# DBTITLE 1,DEFAULT: create_external_endpoint == FALSE
## Setting this to False as default: external endpoint will not be created. 
# Change to True to create external endpoint where pre-requisites are satisfied and corresponding configs are set.

create_external_endpoint = False

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2.2 `if create_external_endpoint == True`

# COMMAND ----------

# MAGIC %md
# MAGIC In the following example code, we will assume that the pre-requisites are satisfied.   
# MAGIC You are welcome to use the code as reference and return to it at a later time.  
# MAGIC
# MAGIC Do review and move on to the subsequent section `2. Deploy the pre-defined model endpoint(s)`. 

# COMMAND ----------

# DBTITLE 1,endpoint parameters & config
if create_external_endpoint:

    external_endpoint_name = f"external_endpt_{endpt_name_suffix}"

    ## AzureOpenai API key in Databricks Secrets
    # SECRETS_SCOPE = "<secrets_scope>"
    # SECRETS_KEY = "AzureOPENAI_API_KEY" # key for AzureOpenAI_API_Token
    # Azure_RESOURCE_NAME | "openai_api_base": "https://<Azure_RESOURCE_NAME>.openai.azure.com/" 

    ext_model_name = "gpt-4o-mini"

    ext_ep_config={                
                    "served_entities": [
                        {
                        "name": external_endpoint_name,
                        "external_model": {
                                            "name": ext_model_name,
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
                        }
                    ],
                    ## Tagging is helpful for: Categorization and Labeling | Cost Monitoring | Lifecycle Management |  Access Control and Environment Management | Custom Identifiers
                    "tags": [
                            {"key": "owner", "value": user_info},
                            {"key": "removeAfter", "value": "2025-03-31"}, # update as appropriate
                            {"key": "tko-demo", "value": "true"}  
                            ],          
                }                

# COMMAND ----------

# ext_ep_config

# COMMAND ----------

# MAGIC %md
# MAGIC ---    
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deploy the pre-defined model endpoint(s) 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Leverage MLflow Deployments
# MAGIC We can use the `mlflow.deployments.get_deploy_client` directly as a `client` to execute the relevant actions:
# MAGIC - [`.create_endpoint()`](https://docs.databricks.com/api/workspace/servingendpoints/create)
# MAGIC - [`.update_endpoint_config()`](https://docs.databricks.com/api/workspace/servingendpoints/updateconfig)
# MAGIC - [`.update_endpoint_tags()`](https://docs.databricks.com/api/workspace/servingendpoints/patch) 
# MAGIC - [`.update_endpoint_ai_gateway()`](https://docs.databricks.com/api/workspace/servingendpoints/putaigateway)
# MAGIC - [`.delete_endpoint()`](https://docs.databricks.com/api/workspace/servingendpoints/delete) 
# MAGIC
# MAGIC Here we create a `class: MLflowDeploymentManager` to wrap these actions as functions with addional logic e.g. for managing endpoint `config_update` where endpoint already exists and to catch thrown errors or exceptions or where there is `RESOURCE_CONFLICTS` etc.

# COMMAND ----------

# DBTITLE 1,Class of functions to create, update & check endpoint status
from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException
from requests.exceptions import HTTPError

class MLflowDeploymentManager:
    """
    A 'custom' manager class to handle MLflow deployment operations such as creating, updating, and deleting endpoints.
    """

    def __init__(self, client_uri="databricks"):
        """
        Initialize the MLflowDeploymentManager with a specified client URI.

        :param client_uri: The URI of the MLflow deployment client. Default is "databricks".
        """
        self.client = get_deploy_client(client_uri)
    
    def get_endpoint(self, ep_name, print_info=False):
        """
        Get the information of an existing endpoint.
        """
        ep_info = self.client.get_endpoint(endpoint=ep_name)
        if print_info:
            print(json.dumps(ep_info, indent=4))
        return ep_info

    def create_endpoint(self, ep_name, ep_config):
        """
        Create a new endpoint with the given name and configuration. If the endpoint already exists, update its configuration.

        :param ep_name: The name of the endpoint to create.
        :param ep_config: The configuration for the endpoint.
        :return: The created or updated endpoint.
        """
        try:
            return self.client.create_endpoint(ep_name, ep_config)
        except (MlflowException, HTTPError) as e:
            if "already exists" in str(e):
                print(f"Endpoint {ep_name} already exists. Updating configuration instead.")
                print('Please be patient -- updates to endpoint config in process... ')                
                return self.update_endpoint_config(ep_name, ep_config)
            else:
                raise e

    def update_endpoint_config(self, ep_name, ep_config):
        """
        Update the configuration of an existing endpoint.

        :param ep_name: The name of the endpoint to update.
        :param ep_config: The new configuration for the endpoint.
        :return: The updated endpoint.
        """
        try:
            return self.client.update_endpoint_config(ep_name, ep_config)
        except MlflowException as e:
            print(f"Failed to update endpoint config for {ep_name}: {e}")
            raise e

    def update_endpoint_tags(self, ep_name, ep_tags):
        """
        Update the tags of an existing endpoint.

        :param ep_name: The name of the endpoint to update.
        :param ep_tags: The new tags for the endpoint.
        :return: The updated endpoint.

        # Example usage 
        # ep_manager.update_endpoint_tags(endpoint_name, ep_config_tags)
        """
        try:
            return self.client.update_endpoint_tags(ep_name, ep_tags)
        except MlflowException as e:
            print(f"Failed to update endpoint tags for {ep_name}: {e}")
            raise e

    def update_endpoint_ai_gateway(self, ep_name, ai_gateway_config):
        """
        Update the AI gateway configuration of an existing endpoint.

        :param ep_name: The name of the endpoint to update.
        :param ai_gateway_config: The new AI gateway configuration for the endpoint.
        :return: The updated endpoint.
        """
        try:
            return self.client.update_endpoint_ai_gateway(ep_name, ai_gateway_config)
        except MlflowException as e:
            print(f"Failed to update AI gateway config for {ep_name}: {e}")
            raise e

    def delete_endpoint(self, ep_name):
        """
        Delete an existing endpoint.

        :param ep_name: The name of the endpoint to delete.
        :return: The result of the delete operation.
        """
        try:
            return self.client.delete_endpoint(ep_name)
        except MlflowException as e:
            print(f"Failed to delete endpoint {ep_name}: {e}")
            raise e

# Instantiate the manager
ep_manager = MLflowDeploymentManager()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Additionally, we also include a function `is_update_in_progress` to check `status` of endpoint `config_update` (e.g. `IN_PROGRESS`) every X seconds until timeout, and to "notify" us when endpoint `config_update` is complete.

# COMMAND ----------

# DBTITLE 1,endpoint config updates check function
import time

def is_update_in_progress(ep_manager, ep_name, check_interval=30, max_retries=10, timeout=None):
    """
    Check if the endpoint configuration update is still in progress.

    :param ep_manager: The MLflowDeploymentManager instance.
    :param ep_name: The name of the endpoint to check.
    :param check_interval: The interval (in seconds) between checks. Default is 30 seconds.
    :param max_retries: The maximum number of retries. Default is 10.
    :param timeout: The maximum time (in seconds) to wait for the update to complete. Default is None (no timeout).
    :return: True if the update is complete, False if it is still in progress or timed out.
    """
    start_time = time.time()
    for _ in range(max_retries):
        ep_details = ep_manager.get_endpoint(ep_name)
        if (ep_details['state']['ready'] == 'READY' and 
            ep_details['state']['config_update'] != "IN_PROGRESS"):
            return True
        print(f"Update in progress for endpoint {ep_name}. Checking again in {check_interval} seconds...")
        time.sleep(check_interval)
        
        if timeout and (time.time() - start_time) > timeout:
            print(f"Timeout reached. Update for endpoint {ep_name} is still in progress.")
            return False
    return False

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Create the provisioned throughput `system.ai` endpoint as defined in `pt_ep_config`

# COMMAND ----------

# DBTITLE 1,create system.ai model  pt endpoint
# Create provisioned throughput endpoint serving system.ai model -- this takes a few minutes to provision

import time
from requests.exceptions import HTTPError

for _ in range(5):
    try:        
        pt_endpoint_details = ep_manager.create_endpoint(pt_endpoint_name, pt_ep_config)
        print('-'*100)       
        # ep_manager.get_endpoint(pt_endpoint_name, prin)

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
# MAGIC #### 2.2.1 While waiting, you can take a look at the Endpoints Serving Tab to locate the `{pt_endpoint_name}` endpoint we just created. 
# MAGIC
# MAGIC The `Serving endpoints` page for the `pt_endpoint_name` we just created will be in the process of updating the configs -- `Pending configuration` as shown -- as well as creating the deployment of the entity to be served. Below is an example of what this looks like. Note that `Serving endpoint state` shows `Not ready (Updating)`: 
# MAGIC
# MAGIC ![01_create_systemai_endpt_pt_pending.png](./imgs/01_create_systemai_endpt_pt_pending.png)   
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.2 When the Deployment is ready the corresponding `Serving endpoint state` will also indicate that it is `Ready`:
# MAGIC
# MAGIC ![01_create_systemai_endpt_pt_ready.png](./imgs/01_create_systemai_endpt_pt_ready.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 [Optional] Create the `external` endpoint as defined  in `ext_ep_config`
# MAGIC
# MAGIC Given that we set `create_external_endpoint = False` (in section 1.2.1 above) the external endpoint will not be created by default.    
# MAGIC Change `create_external_endpoint = True` to create endpoint where pre-requisites are satisfied and corresponding configs are set.
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,create external model endpoint
if create_external_endpoint:
    # Create endpoint serving external model ## typically very fast to provision 
    ext_endpoint_details = ep_manager.create_endpoint(external_endpoint_name, ext_ep_config)

    if is_update_in_progress(ep_manager, external_endpoint_name, timeout=300):
        ep_manager.get_endpoint(external_endpoint_name, print_info=True)
        print(f"Configuration update for endpoint {external_endpoint_name} is complete.")
    else:
        print(f"Configuration update for endpoint {external_endpoint_name} is still in progress or Check time-out reached.")

    ext_endpoint_details
else:
    print("""
        Skipping External Model Endpoint Creation... 
        
        If External Model Endpoint Creation Intended: 
        - Check required pre-requisites and then 
        - Set create_external_endpoint=True to create endpoint
        """
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2.3.1 Example separate test code for creating the `external` endpoint

# COMMAND ----------

# DBTITLE 1,TEST external_endpoint updates
# ## Uncomment below code to execute 

# ext_endpoint_details = ep_manager.create_endpoint(external_endpoint_name, ext_ep_config)

# if is_update_in_progress(ep_manager, external_endpoint_name, timeout=300):
#     ep_manager.get_endpoint(external_endpoint_name, print_info=True)
#     print(f"Configuration update for endpoint {external_endpoint_name} is complete.")
# else:
#     print(f"Configuration update for endpoint {external_endpoint_name} is still in progress or Check time-out reached.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.2 If you have satisfied the pre-requisites to create an `{external}` endpoint, below is an example of what it would look like: 
# MAGIC
# MAGIC ![01_create_external_endpt.png](./imgs/01_create_external_endpt.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Review Served Endpoints
# MAGIC Finally, we can review the `Endpoints Serving Tab` to locate your served endpoint(s); filter by `Created by me` and search for e.g. `endpt_` or other relevant terms. 
# MAGIC
# MAGIC ![01_served_endpoints.png](./imgs/01_served_endpoints.png)

# COMMAND ----------

# MAGIC %md 
# MAGIC ---     

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Endpoints Clean Up
# MAGIC When, at a later stage, the endpoint(s) are no longer needed, it is best to delete them. We will also be doing a workspace clean up at the end. 
# MAGIC
# MAGIC Both `client = mlflow.deployments.get_deploy_client('databricks')` and `ep_manager.delete_endpoint()` can help us achieve this.

# COMMAND ----------

# DBTITLE 1,endpoint(s) clean up
## Delete endpoint when no longer needed 
import mlflow.deployments
client = mlflow.deployments.get_deploy_client('databricks')

# client.delete_endpoint(pt_endpoint_name) ## uncomment to execute
# client.delete_endpoint(external_endpoint_name) ## uncomment to execute

# COMMAND ----------

# MAGIC %md
# MAGIC ## In the following notebooks... 
# MAGIC **_We will walk through how to programmatically `Configure AI Gateway` settings which are currently `Not configured` and illustrate how these settings affect the behaviour of the served endpoint(s)._** 
