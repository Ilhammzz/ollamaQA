"""Chainlit Graph-RAG preparation"""

import os
import re
from typing import Any, Callable, Dict, Tuple
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from ...grag import (
    create_graph_rag_workflow,
    create_graph_visualizer_tool,
    create_hybrid_cypher_retriever_tool,
    create_text2cypher_retriever_tool,
    FallbackToolCalling,
)


URI = os.environ["GRAPH_DATABASE_HOST"]
DATABASE = os.environ["GRAPH_DATABASE_SMALL"]
USERNAME = os.environ["GRAPH_DATABASE_USERNAME"]
PASSWORD = os.environ["GRAPH_DATABASE_PASSWORD"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]


def initialize_graph_rag() -> Tuple[Neo4jGraph, HuggingFaceEmbeddings]:
    """
    TODO: Docstring
    """
    neo4j_graph = Neo4jGraph(
        url=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        enhanced_schema=True,
    )

    embedder_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return neo4j_graph, embedder_model


def configure_graph_rag(
    llm_name: str, neo4j_graph: Neo4jGraph, embedder_model: HuggingFaceEmbeddings
) -> Tuple[BaseChatModel, CompiledStateGraph, Callable[[ToolMessage], Dict[str, Any]]]:
    """
    TODO: Docstring
    """
    neo4j_config = {
        "DATABASE_NAME": DATABASE,
        "ARTICLE_VECTOR_INDEX_NAME": os.environ["ARTICLE_VECTOR_INDEX_NAME"],
        "ARTICLE_FULLTEXT_INDEX_NAME": os.environ["ARTICLE_FULLTEXT_INDEX_NAME"],
        "DEFINITION_VECTOR_INDEX_NAME": os.environ["DEFINITION_VECTOR_INDEX_NAME"],
        "DEFINITION_FULLTEXT_INDEX_NAME": os.environ["DEFINITION_FULLTEXT_INDEX_NAME"],
    }

    neo4j_driver = neo4j_graph._driver

    match = re.search(r"(.*)\/(.*)", llm_name)
    provider, name = match[1], match[2]

    if provider == "google":
        # TODO: Cari konfigurasi paling bagus
        llm = ChatGoogleGenerativeAI(
            model=name,
            temperature=0.0,
            api_key=os.environ["GOOGLE_API_KEY"],
        )
    elif provider == "ollama":
        # TODO: Cari konfigurasi paling bagus
        llm = ChatOllama(
            model=name,
            # num_ctx=16000,
            # num_predict=2048,
            temperature=0.0,
        )

    hybrid_cypher_retriever: Callable[[str], ToolMessage] = (
        create_hybrid_cypher_retriever_tool(
            embedder_model=embedder_model,
            neo4j_driver=neo4j_driver,
            neo4j_config=neo4j_config,
            total_definition_limit=5,
            top_k_initial_article=5,
            max_k_expanded_article=-1,
            total_article_limit=None,
            ranker="linear",
            alpha=0.5,
        )
    )

    text2cypher_retriever: Callable[[str], ToolMessage] = (
        create_text2cypher_retriever_tool(
            neo4j_graph=neo4j_graph,
            embedder_model=embedder_model,
            cypher_llm=llm,
            qa_llm=llm,
            skip_qa_llm=True,
            verbose=False,
        )
    )

    llm_visualizer = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    graph_visualizer_tool: Callable[[ToolMessage], Dict[str, Any]] = (
        create_graph_visualizer_tool(
            llm=llm_visualizer,
            neo4j_graph=neo4j_graph,
            autocomplete_relationship=True,
            verbose=False,
        )
    )

    checkpointer = MemorySaver()

    workflow: CompiledStateGraph = create_graph_rag_workflow(
        model=llm,
        tools=[text2cypher_retriever, hybrid_cypher_retriever],
        checkpointer=checkpointer,
        fallback_tool_calling_cls=FallbackToolCalling,
    )

    return llm, workflow, graph_visualizer_tool
