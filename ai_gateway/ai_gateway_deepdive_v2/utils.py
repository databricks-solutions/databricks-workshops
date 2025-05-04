# Databricks notebook source
# DBTITLE 1,Cluster Config
## Cluster Configuration Used: 
# Runs with standard DBR_15.4LTS (with Dedicated Access | 32-64* GB Memory, 4-8 Cores; *preferred) or Serverless (choose 32Gigs Memory) | r5d.2xlarge | Unity Catalog + Photon

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run Setup... 
# MAGIC `utils.py` includes Helper Functions & Utilites to help set up relevant dependencies and configs for the set of notebooks associated with our **`AI-Gateway Lab`**.

# COMMAND ----------

# DBTITLE 1,Install Latest mlflow dependencies
print("Installing Dependencies... ")

%pip install -q mlflow==2.20.2 #requires a version that supports AI Gateway -- Latest version: mlflow v 2.20.0 as of period of lab development

# %pip install -q --upgrade openai 
%pip install -q openai==1.63.2
# openai v 1.63.2 as of period of lab development
# https://github.com/openai/openai-python/discussions/742


dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,General Configs & Dependencies
# Dependencies
import json
import warnings
import random

# Suppress specific deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*google.protobuf.service module is deprecated.*")


import mlflow
print("mlflow version: ", mlflow.__version__) 

import openai
print("openai version: ", openai.__version__) 
# !openai migrate


# import mlflow.deployments
# client = mlflow.deployments.get_deploy_client("databricks")

# Suppress specific warnings from mlflow.deployments.databricks
# warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.deployments.databricks")

# COMMAND ----------

# DBTITLE 1,user_info
# Get user information from the current session
user_info = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
# print(user_info)

# Create user_info_suffix
# to mitigate overrides in workspaces where similar testing is being done
user_name = user_info.split('@')[0].split('.')
user_seed = sum(ord(char) for name in user_name for char in name) 
random.seed(user_seed)  # Set the seed for reproducibility

if len(user_name[1]) >= 3:
    endpt_name_suffix = ''.join([s[0] for s in user_name[:2]] + [user_name[1][-3]])
else:
    endpt_name_suffix = ''.join([s[0] for s in user_name[:2]] + [str(random.randint(100, 999))])

print(f"user_name: {user_name}")
print(f"endpt_name_suffix: {endpt_name_suffix}")

# COMMAND ----------

# DBTITLE 1,Lab User's Workspace | UC | Endpoint Names
# ## Update to use your own values
# CATALOG_NAME = "<your_UC_catalog_name>"
# SCHEMA_NAME = "<your_UC_schema_name>"

# pt_endpoint_name = f"systemai_endpt_pt_{endpt_name_suffix}" ## endpt_name_suffix variable is derived in ./utils.py
# ENDPOINT_NAME = pt_endpoint_name
# print("ENDPOINT_NAME: ", ENDPOINT_NAME)

# external_endpoint_name = f"external_endpt_{endpt_name_suffix}" ## endpt_name_suffix variable is derived in ./utils.py
# EXT_ENDPOINT_NAME = external_endpoint_name
# print("EXT_ENDPOINT_NAME: ", EXT_ENDPOINT_NAME)

# ## Example AzureOpenai API key in Databricks Secrets
# # SECRETS_SCOPE = "<secrets_scope>"
# # SECRETS_KEY = "AzureOPENAI_API_KEY" # key for AzureOpenAI_API_Token

# ## if you need to add an External API (e.g. AzureOPENAI_API_KEY) key, you can do so with:
# # from databricks.sdk import WorkspaceClient
# # w = WorkspaceClient()
# # w.secrets.put_secret(scope=SECRETS_SCOPE, key=SECRETS_KEY, string_value='<key_value>')

# COMMAND ----------

# DBTITLE 1,MLflowDeploymentManager
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
# ep_manager = MLflowDeploymentManager()

# COMMAND ----------

# DBTITLE 1,is_update_in_progress
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

# Example usage
# endpoint_name = "your_endpoint_name"
# if is_update_in_progress(ep_manager, endpoint_name, timeout=300):
#     print(f"Configuration update for endpoint {endpoint_name} is complete.")
# else:
#     print(f"Configuration update for endpoint {endpoint_name} is still in progress or Check time-out reached.")

# COMMAND ----------

# DBTITLE 1,generate_and_format_queries
import random
from pyspark.sql import functions as F, types as T

def generate_and_format_queries(query_templates, topics, sensitive_words, messageFlag, num_queries, percentage2inject):
    """
    Generates a specified number of random queries based on provided templates and topics, and optionally injects sensitive words into a percentage of these queries.

    Parameters:
    ----------
    query_templates : list
        A list of query templates with placeholders for topics (e.g., "{topic}", "{topic1}", "{topic2}").
    topics : list
        A list of topics to be used to fill in the placeholders in the query templates.
    sensitive_words : list
        A list of sensitive words to be randomly injected into some of the queries.
    messageFlag : str
        A specific string to flag certain queries. Queries containing this flag will not have sensitive words injected.
    num_queries : int
        The number of random queries to generate.
    percentage2inject : float
        The percentage of queries into which sensitive words should be injected.

    Returns:
    -------
    DataFrame
        A Spark DataFrame containing the generated queries, with additional columns indicating whether sensitive words were injected and the modified queries with sensitive words.
    """
    def generate_random_query():
        template = random.choice(query_templates)
        if "{topic1}" in template and "{topic2}" in template:
            return template.format(topic1=random.choice(topics), topic2=random.choice(topics))
        else:
            return template.format(topic=random.choice(topics))

    def generate_queries(num_queries):
        queries = [generate_random_query() for _ in range(num_queries)]
        return queries

    # Generate N queries
    random_queries = generate_queries(num_queries)

    # Create a Spark DataFrame from the random queries
    queries_df = spark.createDataFrame([(messageFlag,)] 
                          + [(query,) for query in random_queries], ["query"]
                         )

    if percentage2inject > 0:
        # Add a strInfo_inject_flag column with X% chance of being True, but always False if query includes messageFlag
        queries_df = queries_df.withColumn(
            "strInfo_inject_flag",
            F.when(F.col("query").contains(messageFlag), F.lit(False))
            .otherwise((F.expr("rand()") < (percentage2inject / 100)).cast("boolean"))  # Change value to change the percentage of queries with strInfo_inject_flag=True
        )

        # Define function to insert 'sensitive words' into Query strings
        def inject_stringInfo(query):
            words = query.split()
            insert_position = random.randint(0, len(words))
            random_string = random.choice(sensitive_words)
            words.insert(insert_position, random_string)
            return " ".join(words)

        inject_stringInfo_udf = F.udf(inject_stringInfo, T.StringType())

        # Add a separate query column to include 'sensitive words' in a random position where strInfo_inject_flag is True
        queries_df = queries_df.withColumn(
            "modified_query",
            F.when(F.col("strInfo_inject_flag"), inject_stringInfo_udf(F.col("query"))).otherwise(F.col("query"))
        )

    return queries_df

## Example usage
# query_templates = [
#     "What is {topic}?",
#     "How does {topic} work?",
#     "Explain the concept of {topic}.",
#     "Compare {topic1} and {topic2}.",
#     "What are the benefits of {topic}?",
#     "Describe the history of {topic}.",
#     "What are the latest trends for {topic}?"
# ]

# topics = ["artificial intelligence", "quantum computing", "blockchain", "machine learning", "cybersecurity", "data science", "travel", "sci-fi", "a protein", "being kind", "being a foodie"]

# sensitive_words = ["pw", "password", "SuperSecretProject"]

# messageFlag = '>>>>>>>>>> SENDING ai_query() requests!!! <<<<<<<<<<<'
# num_queries = 100
# percentage2inject = 5  # ~5% chance to inject sensitive words

# queries_df = generate_and_format_queries(query_templates, topics, sensitive_words, messageFlag, num_queries, percentage2inject)
# display(queries_df)

# COMMAND ----------

# DBTITLE 1,parse_payload
from pyspark.sql import functions as F, types as T

def parse_payload(payload_tablename=None, request_date=F.current_date(), filter_string=''):

    # Define the schema for the request and response fields
    request_schema = T.StructType([
        T.StructField("messages", T.ArrayType(T.StructType([
            T.StructField("role", T.StringType(), True),
            T.StructField("content", T.StringType(), True)
        ])), True)
    ])

    response_schema = T.StructType([
        T.StructField("id", T.StringType(), True),
        T.StructField("object", T.StringType(), True),
        T.StructField("created", T.LongType(), True),
        T.StructField("model", T.StringType(), True),
        T.StructField("choices", T.ArrayType(T.StructType([
            T.StructField("message", T.StructType([
                T.StructField("role", T.StringType(), True),
                T.StructField("content", T.StringType(), True)
            ]), True)
        ])), True),
        T.StructField("usage", T.StructType([
            T.StructField("prompt_tokens", T.IntegerType(), True),
            T.StructField("completion_tokens", T.IntegerType(), True),
            T.StructField("total_tokens", T.IntegerType(), True)
        ]), True)
    ])

    # Load the table into a DataFrame
    df = spark.table(payload_tablename)

    # Filter for dates greater than or equal to the specified date
    filtered_df = df.filter(F.col('request_date') >= request_date)

    # Filter for instances of the specific string in the request JSON
    filtered_instance_df = filtered_df.filter(F.col('request').contains(filter_string))

    # Get the latest request_time for the specific string
    latest_instance_time = filtered_instance_df.agg(F.max('request_time')).collect()[0][0]

    # Filter the original DataFrame to include data after the latest found instance
    filtered_df = df.filter(F.col('request_time') > latest_instance_time)

    # Select and transform the required columns
    df_payload_parsed = filtered_df.select(
        F.col("request_time"),
        F.col("databricks_request_id"),
        F.col("client_request_id"),
        F.col("served_entity_id"),
        F.col("status_code"),
        F.from_json(F.col("request"), request_schema).alias("request_json"),
        F.from_json(F.col("response"), response_schema).alias("response_json")
    ).select(
        F.col("request_time"),
        F.col("databricks_request_id"),
        F.col("client_request_id"),
        F.col("served_entity_id"),
        F.col("status_code"),
        F.regexp_replace(F.col("request_json.messages")[0].content, 'User query:', '').alias("request_messages_user_query"),
        F.col("response_json.id").alias("id"),
        F.col("response_json.object").alias("object"),
        F.col("response_json.created").alias("created"),
        F.col("response_json.model").alias("model"),
        F.col("response_json.choices")[0].message.role.alias("role"),
        F.col("response_json.choices")[0].message.content.alias("response_messages"),
        F.col("response_json.usage.prompt_tokens").alias("prompt_tokens"),
        F.col("response_json.usage.completion_tokens").alias("completion_tokens"),
        F.col("response_json.usage.total_tokens").alias("total_tokens")
    ).dropDuplicates()

    return df_payload_parsed

## Example usage
# payload_tablename = f"{CATALOG_NAME}.{SCHEMA_NAME}.{ENDPOINT_NAME}_payload"
# request_date = F.current_date()
# filter_string = "XXXXXXX Let's start ai_query() requests!!! ---------------------"

# result_df = parse_payload(payload_tablename, request_date, filter_string)
# display(result_df)

# COMMAND ----------

# DBTITLE 1,parse_payload_getEntityName
from pyspark.sql import functions as F, types as T

def parse_payload_getEntityName(payload_tablename, request_date=F.current_date(), filter_string=""):
    """
    Parses the payload from the specified table, filters the data based on the request date and filter string, and extracts relevant fields from the request and response JSON columns.

    Parameters:
    ----------
    payload_tablename : str
        The name of the table containing the payload data.
    request_date : Column, optional
        The date from which to start filtering the data. Default is the current date.
    filter_string : str, optional
        A specific string to filter the request JSON column. Default is an empty string.

    Returns:
    -------
    DataFrame
        A Spark DataFrame with the parsed and filtered payload data, including relevant fields from the request and response JSON columns.
    """
    # Define the schema for the request and response fields
    request_schema = T.StructType([
        T.StructField("messages", T.ArrayType(T.StructType([
            T.StructField("role", T.StringType(), True),
            T.StructField("content", T.StringType(), True)
        ])), True)
    ])

    response_schema = T.StructType([
        T.StructField("id", T.StringType(), True),
        T.StructField("object", T.StringType(), True),
        T.StructField("created", T.LongType(), True),
        T.StructField("model", T.StringType(), True),
        T.StructField("choices", T.ArrayType(T.StructType([
            T.StructField("message", T.StructType([
                T.StructField("role", T.StringType(), True),
                T.StructField("content", T.StringType(), True)
            ]), True)
        ])), True),
        T.StructField("usage", T.StructType([
            T.StructField("prompt_tokens", T.IntegerType(), True),
            T.StructField("completion_tokens", T.IntegerType(), True),
            T.StructField("total_tokens", T.IntegerType(), True)
        ]), True)
    ])

    # Load the table into a DataFrame
    df = spark.table(payload_tablename)

    # Filter for dates greater than or equal to the specified date
    filtered_df = df.filter(F.col('request_date') >= request_date)

    # Filter for instances of the specific string in the request JSON
    filtered_instance_df = filtered_df.filter(F.col('request').contains(filter_string))

    # Get the latest request_time for the specific string
    latest_instance_time = filtered_instance_df.agg(F.max('request_time')).collect()[0][0]

    # Filter the original DataFrame to include data after the latest found instance
    filtered_df = df.filter(F.col('request_time') >= latest_instance_time)

    # Load the served entities table
    df_served_entities = spark.table("system.serving.served_entities")

    # Join the filtered DataFrame with the served entities DataFrame
    joined_df = filtered_df.join(df_served_entities, on="served_entity_id", how="left")

    # Select and transform the required columns
    df_payload_parsed = joined_df.select(
        F.col("request_time"),
        F.col("databricks_request_id"),
        F.col("client_request_id"),
        F.col("served_entity_id"),
        F.col("status_code"),
        F.from_json(F.col("request"), request_schema).alias("request_json"),
        F.from_json(F.col("response"), response_schema).alias("response_json"),
        F.col("entity_name"),
        F.col("endpoint_config_version"),
        F.col("entity_type")
    ).withColumn(
        'model', F.col('entity_name')
    ).select(
        F.col("request_time"),
        F.col("databricks_request_id"),
        F.col("client_request_id"),
        F.col("served_entity_id"),
        F.col("entity_name"),
        F.col("status_code"),
        F.regexp_replace(F.col("request_json.messages")[0].content, 'User query:', '').alias("request_messages_user_query"),
        F.col("response_json.id").alias("id"),
        F.col("response_json.object").alias("object"),
        F.col("response_json.created").alias("created"),
        F.col("model"),  # Use the model column populated with entity_name
        F.col("response_json.choices")[0].message.role.alias("role"),
        F.col("response_json.choices")[0].message.content.alias("response_messages"),
        F.col("response_json.usage.prompt_tokens").alias("prompt_tokens"),
        F.col("response_json.usage.completion_tokens").alias("completion_tokens"),
        F.col("response_json.usage.total_tokens").alias("total_tokens"),
        F.col("endpoint_config_version"),
        F.col("entity_type")
    ).dropDuplicates()

    return df_payload_parsed

## Example usage
# payload_tablename = f"{CATALOG_NAME}.{SCHEMA_NAME}.{ENDPOINT_NAME}_payload"
# request_date = F.current_date()
# filter_string = "XXXXXXX Let's start ai_query() requests!!! ---------------------"

# result_df = parse_payload(payload_tablename, request_date, filter_string)
# display(result_df)

# COMMAND ----------

# DBTITLE 1,parse_payload4queryNresponse
from pyspark.sql import functions as F, types as T
# payload_tablename = f"{CATALOG_NAME}.{SCHEMA_NAME}.{ENDPOINT_NAME}_payload"

def parse_payload4queryNresponse(payload_tablename, request_date=F.current_date(), filter_string=""):
    """
    This function parses the payload from the specified table, filters the data based on the request date and filter string, and extracts relevant fields from the request and response JSON columns. 
    
    The function assumes that the ai_query() is sent with a specific format and prompt structure: "user_prompt >> user_query".

    Example: 

    # Define a prompt template to include '{user_prompt} >> '

      prompt_template = (
      'As a helpful and knowledgeable assistant, you will provide short and very succinct responses. '
      'Return the response as aiq_response key and corresponding value as strings in a {{"aiq_response":"<string>"}} json. '
      'Do not include text before and after the json. >> '
      )

    # Inside the ai_query():

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
        )
    
    Note the 
    to_json(named_struct(
            'userAIQ', array(named_struct('prompt', CONCAT(
              '{prompt_template} >> ', query
            ))),
    
    """
    # Read the data from the table
    df0 = spark.table(payload_tablename)

    # Filter for dates greater than or equal to the specified date
    filterByDate_df = df0.filter(F.col('request_date') >= F.lit(request_date))

    # Filter for instances of the specific string in the request JSON
    filtered_instance_df = filterByDate_df.filter(
        F.col('request').contains(filter_string)
    )

    # Get the latest request_time for the specific string
    latest_instance_time = filtered_instance_df.agg(F.max('request_time')).collect()[0][0]

    # Filter the original DataFrame to include data after the latest found instance
    filtered_df = df0.filter(F.col('request_time') >= latest_instance_time)

    # Define the schema for the request and response JSON columns
    request_schema = T.StructType([
        T.StructField("messages", T.ArrayType(T.StructType([
            T.StructField("content", T.StringType(), True)
        ])), True)
    ])

    response_schema = T.StructType([
        T.StructField("choices", T.ArrayType(T.StructType([
            T.StructField("index", T.IntegerType(), True),
            T.StructField("message", T.StructType([
                T.StructField("role", T.StringType(), True),
                T.StructField("content", T.StringType(), True)
            ]), True),
            T.StructField("finish_reason", T.StringType(), True),
            T.StructField("logprobs", T.StringType(), True)
        ])), True),
        T.StructField("usage", T.StructType([
            T.StructField("prompt_tokens", T.IntegerType(), True),
            T.StructField("completion_tokens", T.IntegerType(), True),
            T.StructField("total_tokens", T.IntegerType(), True)
        ]), True)
    ])

    # Define the schema for the nested JSON within choices.message.content
    nested_response_content_schema = T.StructType([
        T.StructField("aiq_response", T.StringType(), True)
    ])

    # Define the schema for the nested JSON within request.messages.content
    nested_request_content_schema = T.StructType([
        T.StructField("userAIQ", T.ArrayType(T.StructType([
            T.StructField("prompt", T.StringType(), True)
        ])), True),
        T.StructField("max_tokens", T.IntegerType(), True),
        T.StructField("temperature", T.DoubleType(), True)
    ])

    # Parse the request and response columns
    parsed_df = filtered_df.withColumn(
        "parsed_request", F.from_json(F.col("request"), request_schema)
    ).withColumn(
        "parsed_response", F.from_json(F.col("response"), response_schema)
    )

    # Explode the arrays to flatten the structure
    exploded_df = parsed_df.withColumn(
        "message", F.explode_outer(F.col("parsed_request.messages"))
    ).withColumn(
        "choice", F.explode_outer(F.col("parsed_response.choices"))
    )

    # Parse the nested JSON within choices.message.content and request.messages.content
    parsed_content_df = exploded_df.withColumn(
        "parsed_response_content", F.from_json(F.col("choice.message.content"), nested_response_content_schema)
    ).withColumn(
        "parsed_request_content", F.from_json(F.col("message.content"), nested_request_content_schema)
    )

    # Split the prompt into user_prompt and user_query based on ' >> '
    split_prompt_df = parsed_content_df.withColumn(
        "prompt_split", F.split(F.col("parsed_request_content.userAIQ.prompt")[0], r' >> ')
    ).withColumn(
        "user_prompt", F.col("prompt_split")[0]
    ).withColumn(
        "user_query", F.col("prompt_split")[1]
    )

    # Select the required fields
    flattened_df = split_prompt_df.select(
        "request_time",
        "user_prompt",
        "user_query",
        "parsed_request_content.max_tokens",
        "parsed_request_content.temperature",
        "parsed_response_content.aiq_response",
        "parsed_response.usage.prompt_tokens",
        "parsed_response.usage.completion_tokens",
        "parsed_response.usage.total_tokens"
    )

    # Sort the filtered data by request_time in descending order
    return flattened_df.sort('request_time', ascending=False)


## Example usage
# payload_tablename = f"{CATALOG_NAME}.{SCHEMA_NAME}.{ENDPOINT_NAME}_payload"
# request_date = F.current_date()
# filter_string = "XXXXXXX Let's start ai_query() requests!!! ---------------------"

# result_df = parse_payload4queryNresponse(payload_tablename, request_date, filter_string)
# display(result_df)

# COMMAND ----------

# DBTITLE 1,SensitiveKeywordAnalyzer
from pyspark.sql import functions as F

class SensitiveKeywordAnalyzer:
    """
    A class to analyze sensitive keywords in request and response messages within a Spark DataFrame.

    Attributes:
    ----------
    sensitive_keywords : list
        A list of sensitive keywords to search for in the request and response messages.

    Methods:
    -------
    flag_sensitive_keywords(payload_parsed_df):
        Flags the presence of sensitive keywords in the request and response messages and aggregates the results by minute.

    get_sensitive_mentions(payload_parsed_df):
        Retrieves rows where sensitive keywords are mentioned in either the request or response messages, along with the relevant details.
    """

    def __init__(self, sensitive_keywords):
        """
        Initializes the SensitiveKeywordAnalyzer with a list of sensitive keywords.

        Parameters:
        ----------
        sensitive_keywords : list
            A list of sensitive keywords to search for in the request and response messages.
        """
        self.sensitive_keywords = sensitive_keywords

    def flag_sensitive_keywords(self, payload_parsed_df, payload_parserfunc):
        """
        Flags the presence of sensitive keywords in the request and response messages and aggregates the results by minute.

        Parameters:
        ----------
        payload_parsed_df : DataFrame
            A Spark DataFrame containing the parsed payload data with columns 'request_time', 'request_messages_user_query', and 'response_messages'.

        Returns:
        -------
        DataFrame
            A Spark DataFrame with aggregated counts of sensitive keyword mentions in the request and response messages by minute.
        """
        # Create columns to list the sensitive keywords found in the request and response content
        if payload_parserfunc != "parse_payload4queryNresponse" :
            df_flagged = payload_parsed_df.withColumn(
                "Sensitive_Keywords_Request",
                F.expr(f"""
                    filter(
                        array({', '.join([f"'{kw}'" for kw in self.sensitive_keywords])}),
                        kw -> array_contains(split(request_messages_user_query, ' '), kw)
                    )
                """)
            ).withColumn(
                "Sensitive_Keywords_Response",
                F.expr(f"""
                    filter(
                        array({', '.join([f"'{kw}'" for kw in self.sensitive_keywords])}),
                        kw -> array_contains(split(response_messages, ' '), kw)
                    )
                """)
            )

        elif payload_parserfunc == "parse_payload4queryNresponse" :
            df_flagged = payload_parsed_df.withColumn(
                "Sensitive_Keywords_Request",
                F.expr(f"""
                    filter(
                        array({', '.join([f"'{kw}'" for kw in self.sensitive_keywords])}),
                        kw -> array_contains(split(user_query, ' '), kw)
                    )
                """)
            ).withColumn(
                "Sensitive_Keywords_Response",
                F.expr(f"""
                    filter(
                        array({', '.join([f"'{kw}'" for kw in self.sensitive_keywords])}),
                        kw -> array_contains(split(aiq_response, ' '), kw)
                    )
                """)
            )

        # Check for the presence of sensitive keywords in the request and response content
        df_flagged = df_flagged.withColumn(
            "Sensitive_Mention_Request",
            F.when(F.size(F.col("Sensitive_Keywords_Request")) > 0, "Mentioned").otherwise("Not Mentioned")
        ).withColumn(
            "Sensitive_Mention_Response",
            F.when(F.size(F.col("Sensitive_Keywords_Response")) > 0, "Mentioned").otherwise("Not Mentioned")
        )

        # Aggregate the results by minute and mention status for request
        request_counts = df_flagged.groupBy(
            F.col('request_time'),
            F.col('Sensitive_Mention_Request'),
            F.col('Sensitive_Keywords_Request')
        ).count().withColumnRenamed("count", "request_count")

        # Aggregate the results by minute and mention status for response
        response_counts = df_flagged.groupBy(
            F.col('request_time'),
            F.col('Sensitive_Mention_Response'),
            F.col('Sensitive_Keywords_Response')
        ).count().withColumnRenamed("count", "response_count")

        # Join the request and response counts on the minute
        result_df = request_counts.join(
            response_counts,
            on=['request_time'],
            how="left"
        ).orderBy('request_time')

        return result_df

    def get_sensitive_mentions(self, payload_parsed_df, payload_parserfunc, Sensitive_Mention_str2filter):
        """
        Retrieves rows where sensitive keywords are mentioned in either the request or response messages, along with the relevant details.

        Parameters:
        ----------
        payload_parsed_df : DataFrame
            A Spark DataFrame containing the parsed payload data with columns 'request_time', 'request_messages_user_query', and 'response_messages'.

        Returns:
        -------
        DataFrame
            A Spark DataFrame with rows where sensitive keywords are mentioned, including the request and response messages.
        """
        # Create columns to list the sensitive keywords found in the request and response content
        if payload_parserfunc != "parse_payload4queryNresponse" :
            df_flagged = payload_parsed_df.withColumn(
                "Sensitive_Keywords_Request",
                F.expr(f"""
                    filter(
                        array({', '.join([f"'{kw}'" for kw in self.sensitive_keywords])}),
                        kw -> array_contains(split(request_messages_user_query, ' '), kw)
                    )
                """)
            ).withColumn(
                "Sensitive_Keywords_Response",
                F.expr(f"""
                    filter(
                        array({', '.join([f"'{kw}'" for kw in self.sensitive_keywords])}),
                        kw -> array_contains(split(response_messages, ' '), kw)
                    )
                """)
            )
        elif payload_parserfunc == "parse_payload4queryNresponse" :
            df_flagged = payload_parsed_df.withColumn(
                "Sensitive_Keywords_Request",
                F.expr(f"""
                    filter(
                        array({', '.join([f"'{kw}'" for kw in self.sensitive_keywords])}),
                        kw -> array_contains(split(user_query, ' '), kw)
                    )
                """)
            ).withColumn(
                "Sensitive_Keywords_Response",
                F.expr(f"""
                    filter(
                        array({', '.join([f"'{kw}'" for kw in self.sensitive_keywords])}),
                        kw -> array_contains(split(aiq_response, ' '), kw)
                    )
                """)
            )

        # Check for the presence of sensitive keywords in the request and response content
        df_flagged = df_flagged.withColumn(
            "Sensitive_Mention",
            F.when(
                (F.size(F.col("Sensitive_Keywords_Request")) > 0) | (F.size(F.col("Sensitive_Keywords_Response")) > 0),
                "Mentioned"
            ).otherwise("Not Mentioned")
        )

        # Filter for rows where Sensitive_Mention is "Mentioned"
        # df_mentioned = df_flagged.filter(F.col("Sensitive_Mention") == "Mentioned")
        df_mentioned = df_flagged.filter(F.col("Sensitive_Mention") == Sensitive_Mention_str2filter)

        if payload_parserfunc != "parse_payload4queryNresponse" :
            # Select the relevant columns including the request message and response content
            result_df = df_mentioned.select(
                F.date_trunc('MINUTE', F.col('request_time')).alias('minute'),
                F.col('Sensitive_Keywords_Request'),
                F.col('Sensitive_Keywords_Response'),
                F.col('request_messages_user_query').alias('user_query'),
                F.col('response_messages').alias('aiq_response')
            ).orderBy('minute')
        elif payload_parserfunc == "parse_payload4queryNresponse" :
            result_df = df_mentioned.select(
                F.date_trunc('MINUTE', F.col('request_time')).alias('minute'),
                F.col('Sensitive_Keywords_Request'),
                F.col('Sensitive_Keywords_Response'),
                F.col('user_query').alias('user_query'),
                F.col('aiq_response').alias('')
            ).orderBy('minute')

        return result_df

## Example usage
# sensitive_keywords = ["pw", "password", "SuperSecretProject"]
# analyzer = SensitiveKeywordAnalyzer(sensitive_keywords)

# # Flag sensitive keywords
# flagged_df = analyzer.flag_sensitive_keywords(payload_parsed_df, payload_parserfunc)
# display(flagged_df.sort('request_time', ascending=False))

# # Get sensitive mentions
# mentions_df = analyzer.get_sensitive_mentions(payload_parsed_df, payload_parserfunc, Sensitive_Mention_str2filter)
# display(mentions_df.sort('minute', ascending=False))
