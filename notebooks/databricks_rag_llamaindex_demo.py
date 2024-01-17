# Databricks notebook source
# MAGIC %md #Using Databricks Foundation model APIs and Databricks Vector Search with LlamaIndex
# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
dbutils.library.restartPython()  # noqa
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
    endpoint="dbdemos_vs_endpoint",
    index_name="main__build.rag_chatbot_michael_shtelma.databricks_pdf_documentation_self_managed_vs_index",
    host=host,
    token=token,
    text_field="content",
    embedding_field="embedding",
    id_field="id",
)
# dvs.add(docs_list)
dvs_index = VectorStoreIndex.from_vector_store(dvs)
query_engine = dvs_index.as_query_engine()


# COMMAND ----------

# Query and print response
response = query_engine.query("What is Unity Catalog?")
print(response)
