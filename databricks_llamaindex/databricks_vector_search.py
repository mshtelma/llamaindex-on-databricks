from typing import Any, List

from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
)
from llama_index.schema import TextNode, BaseNode

from databricks.vector_search.client import VectorSearchClient


class DatabricksVectorStore(VectorStore):
    def __init__(
        self,
        endpoint: str,
        index_name: str,
        host: str,
        token: str,
        text_field: str,
        embedding_field: str,
        id_field: str,
    ):
        self.endpoint = endpoint
        self.index_name = index_name
        self.host = host
        self.token = token
        self.text_field = text_field
        self.embedding_field = embedding_field
        self.id_field = id_field
        self.vsc = VectorSearchClient(workspace_url=host, personal_access_token=token)
        self.vs_index = self.vsc.get_index(
            endpoint_name=endpoint, index_name=index_name
        )
        self.stores_text = True

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        # MetadataFilters.from_dict({"source": "24"})
        results = self.vs_index.similarity_search(
            columns=[self.text_field, self.id_field],
            query_vector=query.query_embedding,
            num_results=query.similarity_top_k,
            filters=[],
        )
        return VectorStoreQueryResult(
            nodes=[
                TextNode(text=text, id_=id)
                for text, id, score in results["result"]["data_array"]
            ],
            similarities=[score for text, id, score in results["result"]["data_array"]],
            ids=[str(id) for text, id, score in results["result"]["data_array"]],
        )
