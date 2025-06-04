"""Text2Cypher retriever evaluation workflow"""

# import re
import copy
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import ToolCall
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import BasePromptTemplate
from langchain_core.language_models import BaseChatModel
from ragas import EvaluationDataset
from ..retrievers import create_text2cypher_retriever_tool, extract_cypher


def run_text2cypher_workflow(
    evaluation_dataset: EvaluationDataset,
    experiment_name: str,
    *,
    neo4j_graph: Neo4jGraph,
    cypher_llm: BaseChatModel,
    qa_llm: Optional[BaseChatModel] = None,
    # experiment_config: Optional[Dict[str, Any]] = None,
    embedder_model: Optional[Embeddings] = None,
    qa_prompt: Optional[BasePromptTemplate] = None,
    cypher_generation_prompt: Optional[BasePromptTemplate] = None,
    cypher_fix_prompt: Optional[BasePromptTemplate] = None,
    few_shot_prefix_template: Optional[str] = None,
    few_shot_num_examples: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[EvaluationDataset, List[str]]:
    """
    TODO: Docstring
    """
    evaluation_dataset = copy.deepcopy(evaluation_dataset)

    text2cypher_retriever = create_text2cypher_retriever_tool(
        neo4j_graph=neo4j_graph,
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        embedder_model=embedder_model,
        qa_prompt=qa_prompt,
        cypher_generation_prompt=cypher_generation_prompt,
        cypher_fix_prompt=cypher_fix_prompt,
        few_shot_prefix_template=few_shot_prefix_template,
        few_shot_num_examples=few_shot_num_examples,
        add_context_to_artifact=True,
        skip_qa_llm=True,
        verbose=False,
    )

    counter = 0
    generated_cypher_results = []

    for data in tqdm(
        iterable=evaluation_dataset,
        desc=f"Running text2cypher_retriever: `{experiment_name}`",
        disable=not verbose,
    ):
        if counter % 10 == 0 and counter != 0:
            time.sleep(60)

        tool_result = text2cypher_retriever.invoke(
            ToolCall(
                name=text2cypher_retriever.model_dump()["name"],
                args={"query": data.user_input},
                id=f"run-{uuid.uuid4()}-0",  # required
                type="tool_call",  # required
            )
        )

        generated_cypher_results.append(extract_cypher(tool_result.content))

        # retrieved_contexts = [
        #     str(tool_result.artifact["context"])
        # ]

        if tool_result.artifact["is_context_fetched"]:
            retrieved_contexts = []
            for context in tool_result.artifact["context"]:
                retrieved_contexts.append(str(context))
        else:
            # WAJIB, ILHAM JUGA
            retrieved_contexts = [
                "Tidak dapat menemukan data yang sesuai dengan permintaan query"
            ]
        # retrieved_contexts = [
        #     re.search(
        #         r"### \*\*Hasil Eksekusi Kode Cypher ke Database:\*\*\n(.*)",
        #         string=tool_result.content,
        #         flags=re.DOTALL
        #     )[1]
        # ]
        data.retrieved_contexts = retrieved_contexts

        counter += 1

    return evaluation_dataset, generated_cypher_results
