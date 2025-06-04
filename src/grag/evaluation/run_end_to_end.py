"""End-to-end Graph-RAG evaluation workflow"""

import copy
import time
from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from tqdm import tqdm
from IPython.display import display

import langchain_core.messages as langchain_m
from neo4j import RoutingControl
from langchain_neo4j import Neo4jGraph
from langchain_core.tools import BaseTool

# from langchain_core.messages import ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.tool_node import ToolNode

import ragas.messages as ragas_m
from ragas import EvaluationDataset, MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages

from ..workflow import create_graph_rag_workflow
from ..fallback import BaseFallbackToolCalling
from ...prep.encodings import REGULATION_CODES


def _update_multi_turn_user_input(
    messages: List[langchain_m.BaseMessage], multi_turn_sample: MultiTurnSample
) -> None:
    ai_message = None
    tool_messages = []

    for current_message in messages:
        if isinstance(current_message, langchain_m.AIMessage):
            if ai_message is None:
                ai_message = current_message
            else:
                # print(ai_message)
                for tool_call, tool_message in zip(
                    ai_message.tool_calls, tool_messages
                ):
                    multi_turn_sample.user_input.append(
                        ragas_m.AIMessage(
                            content=str(ai_message.content),
                            tool_calls=[
                                ragas_m.ToolCall(
                                    name=tool_call["name"], args=tool_call["args"]
                                )
                            ],
                        )
                    )

                    multi_turn_sample.user_input.append(
                        ragas_m.ToolMessage(content=tool_message.content)
                    )
                ai_message = current_message
                tool_messages = []
        elif isinstance(current_message, langchain_m.ToolMessage):
            tool_messages.append(current_message)
    # For last AIMessagee
    multi_turn_sample.user_input.append(ragas_m.AIMessage(content=ai_message.content))

    # If no tool call and tool message,
    # then create dummy tool call
    if len(multi_turn_sample.user_input) <= 2:
        multi_turn_sample.user_input.insert(
            1,
            ragas_m.AIMessage(
                content="",
                tool_calls=[ragas_m.ToolCall(name="no_tool_call", args={"query": ""})],
            ),
        )
        multi_turn_sample.user_input.insert(2, ragas_m.ToolMessage(content=""))

    # TODO: Bagaimana jika tidak ada tool call?
    # Maka harus buat tool call palsu dengan nama `no_tool_call`


def run_end_to_end_graph_rag_workflow(
    single_turn_evaluation_dataset: EvaluationDataset,
    multi_turn_evaluation_dataset: EvaluationDataset,
    experiment_name: str,
    *,
    model: BaseChatModel,
    neo4j_graph: Neo4jGraph,
    tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode],
    fallback_tool_calling_cls: Optional[Type[BaseFallbackToolCalling]] = None,
    total_definition_limit: int = 5,
    verbose: bool = True,
) -> Tuple[EvaluationDataset, EvaluationDataset]:
    """
    TODO: Docstring
    """
    single_turn_evaluation_dataset = copy.deepcopy(single_turn_evaluation_dataset)
    multi_turn_evaluation_dataset = copy.deepcopy(multi_turn_evaluation_dataset)

    workflow = create_graph_rag_workflow(
        model=model, tools=tools, fallback_tool_calling_cls=fallback_tool_calling_cls
    )

    counter = 0

    for single, multi in tqdm(
        iterable=list(
            zip(single_turn_evaluation_dataset, multi_turn_evaluation_dataset)
        ),
        desc=f"Running end-to-end graph_rag: `{experiment_name}`",
        disable=not verbose,
    ):
        if counter % 10 == 0 and counter != 0:
            time.sleep(60)

        # Run Graph-RAG workflow
        response = workflow.invoke({"messages": single.user_input})
        print("Response Asli LangChain")
        display(response["messages"])
        print()

        _update_multi_turn_user_input(response["messages"], multi)
        print("Response Ragas")
        display(multi.user_input)
        print()

        # response = convert_to_ragas_messages(response["messages"])
        single.response = response["messages"][-1].content
        single.retrieved_contexts = []

        tool_name = ""
        tool_artifact = {}
        # for message in response["messages"]:
        #     if isinstance(message, langchain_m.ToolMessage):
        #         # Only first tool call and first tool message
        #         tool_name = message.name
        #         tool_artifact = message.artifact
        #         break
        tool_messages = [
            (message.name, message.artifact)
            for message in response["messages"]
            if isinstance(message, langchain_m.ToolMessage)
        ]

        if tool_messages:
            for name, artifact in tool_messages:
                if "text2cypher" in name:
                    if artifact["is_context_fetched"]:
                        retrieved_contexts = []
                        for context in artifact["context"]:
                            retrieved_contexts.append(str(context))
                    else:
                        retrieved_contexts = [
                            "Tidak dapat menemukan data yang sesuai dengan permintaan query"
                        ]
                    single.retrieved_contexts += retrieved_contexts
                elif "hybrid" in name:
                    article_node_ids = artifact["node_ids"][:-total_definition_limit]
                    definition_node_ids = artifact["node_ids"][-total_definition_limit:]
                    art_definition_node_ids = []

                    for def_node_id in definition_node_ids:
                        new_definition_id = int(
                            str(def_node_id)[:-6]
                            + REGULATION_CODES["section"]["article"]
                            + "00100"
                        )
                        art_definition_node_ids.append(new_definition_id)

                    query_result = neo4j_graph._driver.execute_query(
                        query_="""
                            UNWIND $node_ids AS node_id
                            MATCH (n)
                            WHERE n.id = node_id
                            RETURN n.text AS text
                        """,
                        parameters_={"node_ids": article_node_ids + art_definition_node_ids},
                        routing_=RoutingControl.READ,
                        database_=neo4j_graph._database,
                    )

                    retrieved_contexts = []
                    for record in query_result.records:
                        retrieved_contexts.append(record["text"])

                    single.retrieved_contexts += retrieved_contexts
                else:
                    # WAJIB, ILHAM JUGA
                    single.retrieved_contexts += [
                        "Tidak dapat menemukan data yang sesuai dengan permintaan query"
                    ]
        else:
            # WAJIB, ILHAM JUGA
            single.retrieved_contexts += [
                "Tidak dapat menemukan data yang sesuai dengan permintaan query"
            ]

        counter += 1

    return single_turn_evaluation_dataset, multi_turn_evaluation_dataset
