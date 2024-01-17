import os
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_completion_callback
from pathlib import Path
from llama_index import download_loader
from llama_index import set_global_service_context

from mlflow.deployments import get_deploy_client

from rag_demo.basic_rag import DatabricksLLM, DatabricksEmbedding
from rag_demo.vectorstore import DatabricksVectorStore


def main_simple_rag():
    # client = get_deploy_client("databricks")
    # print(client.list_endpoints())
    databricks_embedding_model = DatabricksEmbedding()
    service_context = ServiceContext.from_defaults(
        llm=DatabricksLLM(endpoint="databricks-llama-2-70b-chat"),
        embed_model=databricks_embedding_model,
        context_window=2048,
        num_output=256,
    )
    set_global_service_context(service_context)
    PyMuPDFReader = download_loader("PyMuPDFReader")

    loader = PyMuPDFReader()

    input_folder = "/Users/michael.shtelma/Documents/test_pdfs"
    documents = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            documents.extend(
                loader.load_data(
                    file_path=Path(os.path.join(root, file)), metadata=True
                )
            )
    s_index = SummaryIndex.from_documents(documents, service_context=service_context)
    docs_list = list(s_index.docstore.docs.values())
    for doc in docs_list:
        doc.embedding = databricks_embedding_model
    # dvs = DatabricksVectorStore(
    #     "shared-demo-endpoint",
    #     "msh.test.test_direct_vector_index",
    #     "https://e2-demo-field-eng.cloud.databricks.com",
    #     os.environ["DATABRICKS_TOKEN"],
    #     "field2",
    #     "text_vector",
    # )
    dvs = DatabricksVectorStore(
        endpoint="dbdemos_vs_endpoint",
        index_name="main__build.rag_chatbot_michael_shtelma.databricks_pdf_documentation_self_managed_vs_index",
        host="https://e2-demo-field-eng.cloud.databricks.com",
        token=os.environ["DATABRICKS_TOKEN"],
        text_field="content",
        embedding_field="embedding",
        id_field="id",
    )
    # dvs.add(docs_list)
    dvs_index = VectorStoreIndex.from_vector_store(dvs)
    query_engine = dvs_index.as_query_engine()

    # Query and print response
    response = query_engine.query("What is Databricks Delta ?")
    print(response)


if __name__ == "__main__":
    main_simple_rag()
