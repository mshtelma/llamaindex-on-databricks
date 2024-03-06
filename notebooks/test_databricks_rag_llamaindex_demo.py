# Databricks notebook source
# MAGIC %md #Using Databricks Foundation model APIs and Databricks Vector Search with LlamaIndex

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()  # noqa

# COMMAND ----------

from llama_index import ServiceContext, VectorStoreIndex, set_global_service_context

from databricks_llamaindex.databricks_llm import DatabricksLLM, DatabricksEmbedding
from databricks_llamaindex.databricks_vector_search import DatabricksVectorStore

# COMMAND ----------

context = dbutils.entry_point.getDbutils().notebook().getContext()  # noqa
token = context.apiToken().get()
host = context.apiUrl().get()

service_context = ServiceContext.from_defaults(
    llm=DatabricksLLM(endpoint="databricks-llama-2-70b-chat"),
    embed_model=DatabricksEmbedding(endpoint="databricks-bge-large-en"),
    context_window=2048,
    num_output=256,
)
set_global_service_context(service_context)

# COMMAND ----------

dvs = DatabricksVectorStore(
    endpoint="shared-demo-endpoint",
    index_name="souvik_att_demo.default.wiki_articles_demo_bge_index",
    host=host,
    token=token,
    text_field="text",
    embedding_field="__db_text_vector",
    id_field="id",
)
# dvs.add(docs_list)
dvs_index = VectorStoreIndex.from_vector_store(dvs)
query_engine = dvs_index.as_query_engine()


# COMMAND ----------

# Query and print response
response = query_engine.query("Who is Hercules?")
print(response)

# COMMAND ----------


