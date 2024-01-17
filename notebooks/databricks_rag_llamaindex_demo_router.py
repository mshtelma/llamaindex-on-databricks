# Databricks notebook source
# MAGIC %md #Router pattern using LLama Index and Mixtral on Databricks
# MAGIC Adapted tutorial from https://docs.llamaindex.ai/en/stable/examples/low_level/router.html

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
dbutils.library.restartPython()  # noqa

# COMMAND ----------
from typing import List

from pydantic import BaseModel
from pydantic import Field

import json

from llama_index import ServiceContext, VectorStoreIndex, set_global_service_context
from llama_index import PromptTemplate
from llama_index.types import BaseOutputParser
from llama_index.core import BaseQueryEngine
from llama_index.llms import CustomLLM
from llama_index.query_engine import CustomQueryEngine
from llama_index.response_synthesizers import TreeSummarize

from databricks_llamaindex.databricks_llm import DatabricksLLM, DatabricksEmbedding
from databricks_llamaindex.databricks_vector_search import DatabricksVectorStore


# COMMAND ----------


# COMMAND ----------
llm = DatabricksLLM(endpoint="databricks-mixtral-8x7b-instruct")
# COMMAND ----------

choices = [
    "Useful for questions related to Financial Regulation, Capital Requirements Regulation",
    "Useful for questions related to Databricks, Spark",
]


def get_choice_str(choices):
    choices_str = "\n\n".join([f"{idx+1}. {c}" for idx, c in enumerate(choices)])
    return choices_str


choices_str = get_choice_str(choices)
PROMPT_STR = (
    "Some choices are given below. It is provided in a numbered list (1 to"
    " {num_choices}), where each item in the list corresponds to a"
    " summary.\n"
    "---------------------\n{context_list}\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choices"
    " (no more than {max_outputs}, but only select what is needed) that are"
    " most relevant to the question: '{query_str}'\n"
)

router_prompt = PromptTemplate(PROMPT_STR)


def get_formatted_prompt(query_str):
    fmt_prompt = router_prompt.format(
        num_choices=len(choices),
        max_outputs=2,
        context_list=choices_str,
        query_str=query_str,
    )
    return fmt_prompt


query_str = "What instruments are allowed to be used as Common Equity Tier 1 ?"
fmt_prompt = get_formatted_prompt(query_str)
response = llm.complete(fmt_prompt)
print(query_str, "  ", str(response))
query_str = "What benefits does Delta Lake bring over parquet?"
fmt_prompt = get_formatted_prompt(query_str)
response = llm.complete(fmt_prompt)
print(query_str, "  ", str(response))


# COMMAND ----------
class Answer(BaseModel):
    choice: int
    reason: str


FORMAT_STR = """The output should be formatted as a JSON instance that conforms to 
the JSON schema below. 

Here is the output schema:
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "choice": {
        "type": "integer"
      },
      "reason": {
        "type": "string"
      }
    },
    "required": [
      "choice",
      "reason"
    ],
    "additionalProperties": false
  }
}
"""


def _marshal_output_to_json(output: str) -> str:
    output = output.strip()
    left = output.find("[")
    right = output.find("]")
    output = output[left : right + 1]
    return output


class RouterOutputParser(BaseOutputParser):
    def parse(self, output: str) -> List[Answer]:
        """Parse string."""
        json_output = _marshal_output_to_json(output)
        json_dicts = json.loads(json_output)
        answers = [Answer.parse_obj(json_dict) for json_dict in json_dicts]
        return answers

    def format(self, prompt_template: str) -> str:
        return prompt_template + "\n\n" + FORMAT_STR


def route_query(query_str: str, choices: List[str], output_parser: RouterOutputParser):
    fmt_base_prompt = router_prompt.format(
        num_choices=len(choices),
        max_outputs=len(choices),
        context_list=get_choice_str(choices),
        query_str=query_str,
    )
    fmt_json_prompt = output_parser.format(fmt_base_prompt)

    raw_output = llm.complete(fmt_json_prompt)
    parsed = output_parser.parse(str(raw_output))

    return parsed


output_parser = RouterOutputParser()
route_query(
    "What instruments are allowed to be used as Common Equity Tier 1 ?",
    choices,
    output_parser,
)


# COMMAND ----------
class RouterQueryEngine(CustomQueryEngine):
    """Use our Pydantic program to perform routing."""

    query_engines: List[BaseQueryEngine]
    choice_descriptions: List[str]
    verbose: bool = False
    router_prompt: PromptTemplate
    llm: CustomLLM
    summarizer: TreeSummarize = Field(default_factory=TreeSummarize)

    def custom_query(self, query_str: str):
        """Define custom query."""

        # choices_str = get_choice_str(self.choice_descriptions)
        answers = route_query(query_str, choices, output_parser)
        # print choice and reason, and query the underlying engine
        if self.verbose:
            print(f"Selected choice(s):")
            for answer in answers:
                print(f"Choice: {answer.choice}, Reason: {answer.reason}")

        responses = []
        for answer in answers:
            choice_idx = answer.choice - 1
            query_engine = self.query_engines[choice_idx]
            response = query_engine.query(query_str)
            responses.append(response)

        # if a single choice is picked, we can just return that response
        if len(responses) == 1:
            return responses[0]
        else:
            # if multiple choices are picked, we can pick a summarizer
            response_strs = [str(r) for r in responses]
            result_response = self.summarizer.get_response(query_str, response_strs)
            return result_response


# COMMAND ----------

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


finreg_dvs = DatabricksVectorStore(
    endpoint="shared-demo-endpoint",
    index_name="msh.rag_chatbot_msh.databricks_finreg_documentation_self_managed_vs_index",
    host=host,
    token=token,
    text_field="content",
    embedding_field="embedding",
    id_field="id",
)
finreg_index = VectorStoreIndex.from_vector_store(finreg_dvs)
finreg_query_engine = finreg_index.as_query_engine()

spark_dvs = DatabricksVectorStore(
    endpoint="shared-demo-endpoint",
    index_name="msh.rag_chatbot_msh.databricks_pdf_documentation_self_managed_vs_index",
    host=host,
    token=token,
    text_field="content",
    embedding_field="embedding",
    id_field="id",
)
spark_index = VectorStoreIndex.from_vector_store(spark_dvs)
spark_query_engine = spark_index.as_query_engine()

# COMMAND ----------
router_query_engine = RouterQueryEngine(
    query_engines=[finreg_query_engine, spark_query_engine],
    choice_descriptions=choices,
    verbose=True,
    router_prompt=router_prompt,
    llm=llm,
)
response = router_query_engine.query(
    "What reports should a credit institute submit according as part of the Net Stable Funding Ratio module??"
)
print(f"Answer: {response.response}\n\n")
print(f"Used sources ({len(response.source_nodes)}):")
for node in response.source_nodes:
    print(f"Source: {node.text}")
# COMMAND ----------
