from llama_index.embeddings import BaseEmbedding
from llama_index.llms.generic_utils import completion_response_to_chat_response
from pydantic import Extra, Field
from typing import List, Any, Optional, Dict, Sequence

from llama_index import ServiceContext, SimpleDirectoryReader, SummaryIndex
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
)

from llama_index.llms.base import llm_completion_callback, llm_chat_callback

from mlflow.deployments import get_deploy_client

# set context window size
context_window = 2048
# set number of output tokens
num_output = 256


class DatabricksLLM(CustomLLM, extra=Extra.allow):
    endpoint: Optional[str] = "databricks-llama-2-70b-chat"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.client = get_deploy_client("databricks")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=context_window,
            num_output=num_output,
            model_name=self.endpoint,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        result = self.client.predict(
            endpoint=self.endpoint,
            inputs={"messages": [{"role": "user", "content": prompt}]},
        )
        text = result["choices"][0]["message"]["content"]
        return CompletionResponse(text=text.strip(), raw=result)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()


class DatabricksEmbedding(BaseEmbedding, extra=Extra.allow):
    endpoint: Optional[str] = "databricks-bge-large-en"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.client = get_deploy_client("databricks")

    @classmethod
    def class_name(cls) -> str:
        return "DatabricksEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        response = self.client.predict(endpoint=self.endpoint, inputs={"input": query})
        return response["data"][0]["embedding"]

    def _get_text_embedding(self, text: str) -> List[float]:
        response = self.client.predict(endpoint=self.endpoint, inputs={"input": text})
        return response["data"]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(text) for text in texts]
