# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Enable Guardrails for `Sensitive Keys` and `PII`
# MAGIC
# MAGIC The [Mosaic AI Gateway](https://docs.databricks.com/aws/en/ai-gateway) provides several features to handle sensitive keywords and content:
# MAGIC - **Keyword Filtering:** This guardrail allows customers to specify sets of invalid keywords for both input and output. It uses string matching to detect if a keyword exists in the request or response content.
# MAGIC - **Topic Moderation:** Users can define a list of allowed topics. The guardrail flags a chat request if its topic is not among the allowed topics.
# MAGIC - **Safety Filtering:** This feature prevents models from interacting with unsafe and harmful content, such as violent crime, self-harm, and hate speech. It uses `Meta Llama 3's Llama Guard 2-8b` model for content filtering.
# MAGIC - **Personally Identifiable Information (PII) Detection:** The AI Gateway can detect sensitive information like names, addresses, and credit card numbers. It uses [Presidio](https://microsoft.github.io/presidio/supported_entities/) to identify U.S. categories of PII, including credit card numbers, email addresses, phone numbers, bank account numbers, and social security numbers.
# MAGIC
# MAGIC These features help organizations maintain control over AI interactions, ensure compliance with policies, and protect sensitive information in real-time.
# MAGIC
# MAGIC In this notebook, we will review how to configure **`AI Gateway`** to implement guardrails for sensitive/invalid **`Keyword Filtering`** and **`(PII) Detection`**, with the view that the process of configuring other settings can be extrapolated as needed.

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
# Example workspace Personal Access Token or Service Principal API key in Databricks Secrets
DATABRICKS_SECRETS_SCOPE = "<workspace_secrets_scope>" # scope name associated to secret_key where its key_value is stored
DATABRICKS_SECRETS_KEY = "<databricks_{PAT/SP}token>" # key for workspace Personal Access Token (PAT) / Service Principal (SP)

DATABRICKS_API_TOKEN = dbutils.secrets.get(DATABRICKS_SECRETS_SCOPE, DATABRICKS_SECRETS_KEY) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Review Inference `payload` for presence of `sensitive keywords`
# MAGIC
# MAGIC We have seen in an earlier notebook that when we inject "sensitive keywords" into the query some of the responses also include these sensitive words. 
# MAGIC
# MAGIC Let's review our Inference **`payload`** and make a general assessment of the frequency of such sensitive words injection in both requests and responses. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Parse `payload`
# MAGIC
# MAGIC We will use  our **`messageFlag`** as `timestamp` `marker` to filter out inference table payload for the most recent bulk of sent prompt/queries before running the the filtered `payload` data with the _**helper_function**_ **`parse_payload()`**.  

# COMMAND ----------

# DBTITLE 1,payload_parsed
payload_tablename = f"{CATALOG_NAME}.{SCHEMA_NAME}.{ENDPOINT_NAME}_payload"

request_date = "2025-02-20" # F.current_date() # "yyyy-mm-dd" -- is the default value | Please update to use the date that you ran notebook 02_AIGateway_Enable_UsageInference and sent a bunch of queries where some included sensitive information. 

messageFlag = '>>>>>>>>>> SENDING ai_query() <<<strInject>>> requests!!! <<<<<<<<<<<'
filter_string = messageFlag

payload_parsed_df = parse_payload4queryNresponse(payload_tablename, request_date, filter_string)
payload_parsed_df = payload_parsed_df.filter(F.col('aiq_response').isNotNull())
display(payload_parsed_df.sort('request_time', ascending=False) )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Flag sensitive key words 
# MAGIC
# MAGIC We can get the help of our _**helper_class**_ **`SensitiveKeywordAnalyzer`** to further flag the presence (`flag_sensitive_keywords()`) of sensitive keywords and their mentions (`get_sensitive_mentions()`). 

# COMMAND ----------

# DBTITLE 1,Flag sensitive keywords in Batch of requests (output hidden)
sensitive_keywords = ["pw", "password", "SuperSecretProject"]
analyzer = SensitiveKeywordAnalyzer(sensitive_keywords)

# Flag sensitive keywords across the Batch of requests sent for Inference requests
flagged_df = analyzer.flag_sensitive_keywords(payload_parsed_df, "parse_payload4queryNresponse")

# display(flagged_df.sort('request_time', ascending=False)) 
display(flagged_df.sort('Sensitive_Mention_Request','request_time', ascending=True)) 
# Display of output flagged_df is hidden; you can show results to explore if interested 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Check for sensitive keywords in `responses` of known `prompt-injected` `queries` 
# MAGIC Focusing on those queries that we actually intentionally **prompt-injected** those sensitive words, it is clear that **a portion of the responses were 'injected' with one of the sensitive terms.** 
# MAGIC
# MAGIC Just to be conservative, we will filter for all the sensitive keyword mentions in both requests and responses and also their corresponding content. 

# COMMAND ----------

# DBTITLE 1,known prompt-injected queries and corresponding response
# Get sensitive mentions
mentions_df = analyzer.get_sensitive_mentions(payload_parsed_df, "parse_payload4queryNresponse", "Mentioned")
display(mentions_df.sort('minute','Sensitive_Keywords_Request', ascending=True))

# COMMAND ----------

# MAGIC %md
# MAGIC In this filtered analysis here, the goal is to **investigate if `prompt-injection` existed in the `request` whether the `sensitive keyword` would also be found in `response`**.
# MAGIC
# MAGIC We see that prompt-injection of "`SuperSecretProject`" and "`password`" had a higher occurence of being found in the response, whereas "`pw`" had only 1 case of response-injection. 
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure Endpoint Guardrails for `Invalid Keywords` with AI Gateway
# MAGIC
# MAGIC To _prevent the endpoint from being used for inappropriate / out-of-scope topics_ such as those in the list of sensitive key terms, we can configure our endpoint to include `input` `Guardrails` for `invalid keywords` of interest. 
# MAGIC
# MAGIC NB: There is also the option to configure `output` `Guardrails` for `invalid keywords` of interest to safeguard transmission of sensitive info. in responses.  
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Endpoint `ai_gateway` Config. Update Process: [`Invalid keys`] `guardrials` 

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
# MAGIC
# MAGIC Let's check the `ai_gateway` configurations of our existing provisisioned throughput endpoint `systemai_endpt_pt_{}`:
# MAGIC
# MAGIC We have so far enabled `usage_tracking_config`, `inference_table_config`, and `rate_limits`. 
# MAGIC
# MAGIC There are currently neither `guardrails` for `invalid_keywords` nor  `PII` are enabled.   
# MAGIC
# MAGIC Let's start with configuring `guardrails[invalid_keywords]` settings.

# COMMAND ----------

# DBTITLE 1,existing ai_gateway configs
## we can use the client to get the existing config and update it
ep_info = ep_manager.get_endpoint(ENDPOINT_NAME)
pt_ep_config_aigateway = ep_info['ai_gateway']  
pt_ep_config_aigateway

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.3 Specify Endpoint `ai_gateway` settings for desired `guardrails` `input["invalid_keywords"]`
# MAGIC
# MAGIC We first update the endpoint `ai_gateway` config dictionary with the desired `input["invalid_keywords"]`.    
# MAGIC You can provide a list of keywords, which is what we illustrate here: 

# COMMAND ----------

# DBTITLE 1,update ai_gateway configs
import json

pt_ep_config_aigateway.update({
                            "guardrails": {
                                "input": {
                                    "invalid_keywords": ["pw", "password", "SuperSecretProject"],
                                },
                            }
                        })

# check
pt_ep_config_aigateway

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.4 Push `ai_gateway` configuration with `guardrails` updates to endpoint
# MAGIC
# MAGIC With the updated `ai_gateway` configurations defined, we can use our instantiated endpoint manager client (`ep_manager`) to push the updated `ai_gateway` configuration settings on the endpoint: 

# COMMAND ----------

# DBTITLE 1,Client Update Endpoint AI Gateway Configs
# Update AI Gateway Configs using the MLflowDeploymentManager
ep_manager.update_endpoint_ai_gateway(pt_endpoint_name, pt_ep_config_aigateway)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.5 Review the updated `ai_gateway` configuration for `guardrails`
# MAGIC
# MAGIC We can call a `get_endpoint` info. to review the updated configuration settings and verify that the update was successful.    
# MAGIC
# MAGIC The `invalid_keywords` settings we specified should show up within the `ai_gateway` key-value in the configuration dictionary.  

# COMMAND ----------

ep_manager.get_endpoint(pt_endpoint_name)

# COMMAND ----------

# DBTITLE 1,GET UI screenshot for updates
# MAGIC %md
# MAGIC #### 2.1.6 Review Serving Endpoint UI for updated `ai_gateway` configs 
# MAGIC
# MAGIC Once we enabled, the `guardrails` `input` configuration settings show `3 invalid keywords` on the `Serving endpoints UI` for `Gateway`. 
# MAGIC
# MAGIC
# MAGIC ![04_Guardrails_Invalid_Keywords.png](./imgs/04_Guardrails_Invalid_Keywords.png)
# MAGIC
# MAGIC ![]() 
# MAGIC
# MAGIC Click to `edit` and you will see the actual `Invalid keywords for input` listed.   
# MAGIC
# MAGIC ![04_Guardrails_Invalid_Keywords_inputDetails.png](./imgs/04_Guardrails_Invalid_Keywords_inputDetails.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Validate that Enabled Guardrails for `Invalid keywords` is working as intended on Endpoint AI-Gateway    
# MAGIC
# MAGIC Let's check on the enabled `Sensitive/Invalid keywords` `Guardrail` to see it in action. 
# MAGIC
# MAGIC We will need to simluate some prompts for the testing:

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.1 Create some prompts to use for testing `Sensitive/Invalid keywords` `Guardrail`

# COMMAND ----------

# DBTITLE 1,TEST if sensitive keywords are activated in guardrails
prompts2test = ["How do I reset my pw for the SuperSecretProject?", # keywords
                "Tell me about Quantum Computing", 
                "I forgot my password for the SuperSecretProject, can you help?", # keywords
                "Examples of Sensitive keywords for AI Gateway",
                "Is there a way to recover the pw for the SuperSecretProject?", # keywords
                "Can you tell me the password for the SuperSecretProject?", # keywords
                "What is the difference between Quantum and neuromorphic computing?",
                "I need the pw to access the SuperSecretProject files.", # keywords
                "What is the default password for the SuperSecretProject?", # keywords
                "I need help with PW and explore our SuperSecretProject quantum computing research." # keywords
                ] 

# Example usage
# prompt = random.choice(prompts2test)                

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.2 Leverage `OpenAI Python Client`
# MAGIC We will use i) the [OpenAI Python Client](https://github.com/openai/openai-python) to help test our randomly sampled pseudo queries with or without `invalid_keywords[sensitive keywords]` and ii) a wrapper function to `test_InvalidKeys_guardrails()`

# COMMAND ----------

# DBTITLE 1,OpenAI client settings
from openai import OpenAI

# We will use PAT (Personal Access Token) or Service Principal Token for the API key required to access the REST AP
# If not already defined above
# DATABRICKS_API_TOKEN = dbutils.secrets.get(DATABRICKS_SECRETS_SCOPE, DATABRICKS_SECRETS_KEY)

# Retrieve the workspace host URL from the Databricks notebook context
DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)

# Query the endpoint with OpenAI client 
client = OpenAI(
   api_key=DATABRICKS_API_TOKEN,
   base_url=f"{DATABRICKS_HOST}/serving-endpoints",
)

# COMMAND ----------

# MAGIC %md 
# MAGIC We will also define a wrapper function `test_InvalidKeys_guardrails()` for the client to test the guardrails-enabled `AI Gateway`. 

# COMMAND ----------

# DBTITLE 1,Function to test Invalid keywords guardrails
def test_InvalidKeys_guardrails(served_endpoint_name = ENDPOINT_NAME, prompt = None):

  try:
      response = client.chat.completions.create(
      model=served_endpoint_name,
      messages=[
          {"role": "system", "content": "You are a helpful assistant. Please be succinct in your response."},
          {"role": "user", "content": prompt},
      ],
      max_tokens=500, #
      timeout=30  # Add a timeout of 30 seconds
      )

      print('Prompt Sent: ', prompt)
      print('-'*100)
      print(response.choices[0].message.content)
      
  except Exception as e:
      if "invalid_keywords" in str(e):
        print("Error: Invalid keywords detected in the prompt. Please revise your input.")
        print('-'*100)
        print("Prompt Sent: ", prompt)
        print('-'*100)
        print(e)      
      else:
        print(f"An error occurred: {e}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.3 Test `Invalid keywords` guardrails 
# MAGIC
# MAGIC Now we are ready to test to see if our updated `AI-Gateway` Settings will `BLOCK` the `Invalid keywords`(s) it detects.
# MAGIC
# MAGIC Example test outputs :
# MAGIC ```
# MAGIC Error: Invalid keywords detected in the prompt. Please revise your input.
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC Prompt Sent:  I forgot my password for the SuperSecretProject, can you help?
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC Error code: 400 - {'usage': {'prompt_tokens': 0, 'total_tokens': 0}, 'input_guardrail': [{'flagged': False, 'categories': None, 'category_scores': None, 'pii_detection': None, 'anonymized_input': None, 'invalid_keywords': True, 'off_topic': None}], 'finishReason': 'input_guardrail_triggered'}
# MAGIC ```
# MAGIC ----------------------------------------------------------------------------------------------------   
# MAGIC
# MAGIC ~~~
# MAGIC
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC ```
# MAGIC Error: Invalid keywords detected in the prompt. Please revise your input.
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC Prompt Sent:  I need help with PW and explore our SuperSecretProject quantum computing research.
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC Error code: 400 - {'usage': {'prompt_tokens': 0, 'total_tokens': 0}, 'input_guardrail': [{'flagged': False, 'categories': None, 'category_scores': None, 'pii_detection': None, 'anonymized_input': None, 'invalid_keywords': True, 'off_topic': None}], 'finishReason': 'input_guardrail_triggered'}
# MAGIC ```
# MAGIC
# MAGIC **NB:** _It may take a few minutes for the endpoint configurations to update and for the guardrails to kick in; please try running the cell again if the response seem unexpected._  

# COMMAND ----------

# DBTITLE 1,TEST example with Sensitive Keys
test_InvalidKeys_guardrails(served_endpoint_name = ENDPOINT_NAME, prompt = random.choice(prompts2test))

# COMMAND ----------

# DBTITLE 1,TEST example with Sensitive Keys
test_InvalidKeys_guardrails(served_endpoint_name = ENDPOINT_NAME, prompt = random.choice(prompts2test))

# COMMAND ----------

# DBTITLE 1,TEST example with{out} Sensitive Keys
test_InvalidKeys_guardrails(served_endpoint_name = ENDPOINT_NAME, prompt = "Examples of Sensitive keywords for AI Gateway")

# COMMAND ----------

# MAGIC %md
# MAGIC We see that our updated AI Gateway Guardrails config kicked in for `sensitive/invalid_keywords` and was able to filter out sensitive keywords in the prompts. 

# COMMAND ----------

# MAGIC %md
# MAGIC ---    

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure Endpoint Guardrails for `PII` with AI Gateway 
# MAGIC
# MAGIC Now let's configure guardrails for **`PII`** since it is just as important to prevent access and exposure to such information both from the `query input` as well as the `response output`. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Endpoint `ai_gateway` Config. Update Process: [`PII`]

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.1 Exisiting Endpoint `ai_gateway` Config: `pt_ep_config_aigateway`
# MAGIC
# MAGIC Once again we will check our current endpoint configurations: 
# MAGIC
# MAGIC We have so far enabled `usage_tracking_config`, `inference_table_config`, `rate_limits`, and `invalid_keywords`.
# MAGIC
# MAGIC There are currently no `guardrails` for `PII` enabled, and also no sign of any `guardrails[input/output]` settings.   
# MAGIC
# MAGIC Let's go ahead and add them. 

# COMMAND ----------

# DBTITLE 1,Retrieve existing AI Gateway Configs
## we can use the client to get the existing config and update it
ep_info = ep_manager.get_endpoint(ENDPOINT_NAME)
pt_ep_config_aigateway = ep_info['ai_gateway']  
pt_ep_config_aigateway

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.2 Specify Endpoint `ai_gateway` behavior for desired `guardrails` `input["PII"]`
# MAGIC
# MAGIC We first update the endpoint `ai_gateway` config dictionary with the desired `["PII"]` behavior, ie to `BLOCK` both `input` and `output` with detected `PII` information. 

# COMMAND ----------

# DBTITLE 1,Update exsiting AI Gateway Configs
pt_ep_config_aigateway.update({
                            "guardrails": {
                                "input": {
                                          "pii": {"behavior": "BLOCK"}, # other behavior options include "MASK"
                                          "invalid_keywords": ["SuperSecretProject","pw","password"], ## include this again to avoid overwrite with only "pii" 'updates'                                        
                                      },
                                      "output": {
                                          "pii": {"behavior": "BLOCK"}, # other behavior options include "MASK"
                                      },
                            }
                        })

# check
pt_ep_config_aigateway                            

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.3 Push `ai_gateway` configuration with `guardrails` updates to endpoint
# MAGIC
# MAGIC With the updated `ai_gateway` configurations defined, we can use our instantiated endpoint manager client (`ep_manager`) to push the updated `ai_gateway` configuration settings on the endpoint: 

# COMMAND ----------

# DBTITLE 1,Client Update Endpoint AI Gateway Configs
# Update AI Gateway Configs using the MLflowDeploymentManager
ep_manager.update_endpoint_ai_gateway(pt_endpoint_name, pt_ep_config_aigateway)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.4 Review the updated `ai_gateway` configuration for `guardrails`
# MAGIC
# MAGIC We can call a `get_endpoint` info. to review the updated configuration settings and verify that the update was successful. The enabled `PII` settings we specified should show up within the `ai_gateway` key-value in the configuration dictionary.  

# COMMAND ----------

ep_manager.get_endpoint(pt_endpoint_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 3.1.5 Review Serving Endpoint UI for updated `ai_gateway` configs 
# MAGIC
# MAGIC The updated AI Gateway Config. with `PII` enabled should also manifests in the `Serving endpoints UI`
# MAGIC
# MAGIC ![04_Guardrails_PII.png](./imgs/04_Guardrails_PII.png)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Test Endpoint Gateway `PII` Detection
# MAGIC
# MAGIC Now that we have enabled **`AI Gatway Guardrails`** to monitor and **`detect PII`** we will run a few tests to _**check that it is activated when potential PII is detected**_. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2.1 Generate `PII/nonPII` pseudo queries
# MAGIC
# MAGIC We will generate some pseudo queries that either include employee data or has suggestive queries to inquire about staff info. and also mix in some that don't inquire or reference any such PIIs. 

# COMMAND ----------

# DBTITLE 1,Generate Random PII / non-PII Prompts
import random

# Set a seed for reproducibility
random.seed(202503)

# Generate random fictional data
names = ["Samantha Lee", "Raj Patel", "Elena Rodriguez", "John Doe", "Jane Smith"]
emails = ["slee@fictional-corp.com", "rpatel@imaginary-tech.net", "erodriguez@pretend-company.org", "jdoe@fake-email.com", "jsmith@mock-domain.com"]
phones = ["(555) 123-4567", "(555) 987-6543", "(555) 246-8135", "(555) 654-3210", "(555) 789-0123"]
titles = ["Senior Marketing Manager", "Software Engineer II", "Director of Operations", "Product Manager", "Data Scientist"]

def generate_fictional_data(names, emails, phones, titles):    
    data = []
    for name, email, phone, title in zip(names, emails, phones, titles):
        data.append(f"{name}, {email}, {phone}, {title}")
    return "\n".join(data)

# Example usage
# fictional_data = generate_fictional_data()

# List of prompts with PII
pii_prompts = [
    f"What is {random.choice(names)}'s phone number and email address?\n\nEmployee data:\n{generate_fictional_data(names, emails, phones, titles)}",
    f"What is {random.choice(names)}'s job title?\n\nEmployee data:\n{generate_fictional_data(names, emails, phones, titles)}",
    f"Can you provide the email address of {random.choice(names)}?\n\nEmployee data:\n{generate_fictional_data(names, emails, phones, titles)}",
    f"Who is the {random.choice(titles)}?\n\nEmployee data:\n{generate_fictional_data(names, emails, phones, titles)}",
    f"What is the phone number of {random.choice(names)}?\n\nEmployee data:\n{generate_fictional_data(names, emails, phones, titles)}"
]

# List of prompts without PII
non_pii_prompts = [
    f"What are the responsibilities of a {random.choice(titles)}?",
    f"Describe the role of a {random.choice(titles)}.",
    f"What does a {random.choice(titles)} do?",
    f"What are the key skills required for a {random.choice(titles)}?",
    f"What is the job description of a {random.choice(titles)}?"
]

## Combine both lists and randomly select a prompt
all_prompts = pii_prompts + non_pii_prompts
# prompt = random.choice(all_prompts)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2.2 Test pseudo prompts on guardrails for `PII/nonPII` 
# MAGIC We will once again use the [OpenAI Python Client](https://github.com/openai/openai-python) to help test our randomly sampled `PII/nonPII` pseudo queries 

# COMMAND ----------

# DBTITLE 1,OpenAI client settings
## Ought to have been already instantiated above

# from openai import OpenAI

## Retrieve the internal token and host URL from the Databricks notebook context
# If not already defined above
# DATABRICKS_API_TOKEN = dbutils.secrets.get(DATABRICKS_SECRETS_SCOPE, DATABRICKS_SECRETS_KEY)
# DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)

# # Query the endpoint with OpenAI client 
# client = OpenAI(
#    api_key=DATABRICKS_API_TOKEN,
#    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
# )

# COMMAND ----------

# MAGIC %md
# MAGIC We will also define a wrapper function `test_PII_guardrails()` to call the client API and send our PII-related prompts:

# COMMAND ----------

# DBTITLE 1,Function to test_PII_guardrails
def test_PII_guardrails(served_endpoint_name = ENDPOINT_NAME, prompt = None):

  try:
      response = client.chat.completions.create(
      model=served_endpoint_name,
      messages=[
          {"role": "system", "content": "You are a helpful assistant. Please be succinct in your response."},
          {"role": "user", "content": prompt},
      ],
      max_tokens=500,
      timeout=30  # Add a timeout of 30 seconds
      )

      print('Prompt Sent: ', prompt)
      print('-'*100)
      print(response.choices[0].message.content)

  except Exception as e:
      if "pii_detection" in str(e):
          print('Prompt Sent: ', prompt)
          print('-'*100)
          print("Error: PII (Personally Identifiable Information) detected. Please try again.")
          print('-'*100)
          print(e)
      else:
          print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 3.2.3 Test pseudo prompts on guardrails for `PII` 
# MAGIC
# MAGIC Below are examples of our **`AI Gateway Guardrial enabled for PII`** kicking in. 
# MAGIC
# MAGIC -----    
# MAGIC
# MAGIC ```
# MAGIC Prompt Sent:  Who is the Data Scientist?
# MAGIC
# MAGIC Employee data:
# MAGIC Samantha Lee, slee@fictional-corp.com, (555) 123-4567, Senior Marketing Manager
# MAGIC Raj Patel, rpatel@imaginary-tech.net, (555) 987-6543, Software Engineer II
# MAGIC Elena Rodriguez, erodriguez@pretend-company.org, (555) 246-8135, Director of Operations
# MAGIC John Doe, jdoe@fake-email.com, (555) 654-3210, Product Manager
# MAGIC Jane Smith, jsmith@mock-domain.com, (555) 789-0123, Data Scientist
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC Error: PII (Personally Identifiable Information) detected. Please try again.
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC Error code: 400 - {'usage': {'prompt_tokens': 0, 'total_tokens': 0}, 'input_guardrail': [{'flagged': False, 'categories': None, 'category_scores': None, 'pii_detection': True, 'anonymized_input': [{'role': 'system', 'content': 'You are a helpful assistant. Please be succinct in your response.'}, {'role': 'user', 'content': 'Who is the Data Scientist?\n\nEmployee data:\nSamantha Lee, <EMAIL_ADDRESS>, <PHONE_NUMBER>, Senior Marketing Manager\nRaj Patel, <EMAIL_ADDRESS>, <PHONE_NUMBER>, Software Engineer II\nElena Rodriguez, <EMAIL_ADDRESS>, <PHONE_NUMBER>, Director of Operations\nJohn Doe, <EMAIL_ADDRESS>, <PHONE_NUMBER>, Product Manager\nJane Smith, <EMAIL_ADDRESS>, <PHONE_NUMBER>, Data Scientist'}], 'invalid_keywords': False, 'off_topic': None}], 'finishReason': 'input_guardrail_triggered'}
# MAGIC
# MAGIC ```
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC
# MAGIC ~~~    
# MAGIC
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC
# MAGIC ```
# MAGIC Prompt Sent:  What is the phone number of Elena Rodriguez?
# MAGIC
# MAGIC Employee data:
# MAGIC Samantha Lee, slee@fictional-corp.com, (555) 123-4567, Senior Marketing Manager
# MAGIC Raj Patel, rpatel@imaginary-tech.net, (555) 987-6543, Software Engineer II
# MAGIC Elena Rodriguez, erodriguez@pretend-company.org, (555) 246-8135, Director of Operations
# MAGIC John Doe, jdoe@fake-email.com, (555) 654-3210, Product Manager
# MAGIC Jane Smith, jsmith@mock-domain.com, (555) 789-0123, Data Scientist
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC Error: PII (Personally Identifiable Information) detected. Please try again.
# MAGIC ----------------------------------------------------------------------------------------------------
# MAGIC Error code: 400 - {'usage': {'prompt_tokens': 0, 'total_tokens': 0}, 'input_guardrail': [{'flagged': False, 'categories': None, 'category_scores': None, 'pii_detection': True, 'anonymized_input': [{'role': 'system', 'content': 'You are a helpful assistant. Please be succinct in your response.'}, {'role': 'user', 'content': 'What is the phone number of Elena Rodriguez?\n\nEmployee data:\nSamantha Lee, <EMAIL_ADDRESS>, <PHONE_NUMBER>, Senior Marketing Manager\nRaj Patel, <EMAIL_ADDRESS>, <PHONE_NUMBER>, Software Engineer II\nElena Rodriguez, <EMAIL_ADDRESS>, <PHONE_NUMBER>, Director of Operations\nJohn Doe, <EMAIL_ADDRESS>, <PHONE_NUMBER>, Product Manager\nJane Smith, <EMAIL_ADDRESS>, <PHONE_NUMBER>, Data Scientist'}], 'invalid_keywords': False, 'off_topic': None}], 'finishReason': 'input_guardrail_triggered'}
# MAGIC ```
# MAGIC
# MAGIC
# MAGIC -----     
# MAGIC
# MAGIC _**Please Note:**_
# MAGIC
# MAGIC - The first execution of the `test_PII_guardrails(ENDPOINT_NAME, prompt=random.choice(all_prompts))` function can take a while to run if the endpoint is `scaled-to-zero` as it requires 'warming up!'
# MAGIC
# MAGIC - If you test it too soon, you may not see that `PII` is being detected yet. Give it another couple of minutes before testing again.  
# MAGIC
# MAGIC - Test and check the guardrail responses prior to productionization. It is possbile that on occasion  response could be 'missed' 

# COMMAND ----------

# DBTITLE 1,PII detection
test_PII_guardrails(ENDPOINT_NAME, prompt=random.choice(pii_prompts))

# COMMAND ----------

# DBTITLE 1,PII detection
test_PII_guardrails(ENDPOINT_NAME, prompt=random.choice(pii_prompts))

# COMMAND ----------

# DBTITLE 1,EXAMPLE PII detection is able to differentiate Query
test_PII_guardrails(ENDPOINT_NAME, prompt=random.choice(non_pii_prompts))

# COMMAND ----------

# DBTITLE 1,test_combined_all_prompts
test_PII_guardrails(ENDPOINT_NAME, prompt=random.choice(all_prompts))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. NEXT
# MAGIC
# MAGIC Looks like the `guardrails` we enabled via the `AI-Gateway` are working as intended. 
# MAGIC
# MAGIC With the **AI Gateway** configuration updates to our endpoint for **`Sensitive/Invalid keywords`** and **`PII detection`** now in place, we will move on to learn how we can perform `Model Comparison` and associated endpoint traffic routing with **`A/B Testing`**.
