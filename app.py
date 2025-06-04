"""Chainlit app: ```chainlit run app.py```"""

import chainlit as cl
from dotenv import load_dotenv
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableConfig
from src.ui import (
    GRAPH_RAG_DESC,
    GRAPH_RAG_SETTINGS,
    GRAPH_RAG_STARTERS,
    configure_graph_rag,
    graph_rag_on_message,
    initialize_graph_rag,
)


@cl.set_chat_profiles
async def chat_profile():
    """
    TODO: Docstring
    """
    return [
        cl.ChatProfile(
            name="Graph-RAG",
            markdown_description=GRAPH_RAG_DESC,
            icon="https://picsum.photos/200",
            starters=GRAPH_RAG_STARTERS,
        ),
        # TODO
        cl.ChatProfile(
            name="TAG",
            markdown_description="TODO",
            icon="https://picsum.photos/250",
            starters=None,
            # default=True
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """
    TODO: Docstring
    """
    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "Graph-RAG":
        if not cl.user_session.get("is_graph_rag_initialized ", False):
            neo4j_graph, embedder_model = initialize_graph_rag()

            cl.user_session.set("neo4j_graph", neo4j_graph)
            cl.user_session.set("embedder_model", embedder_model)
            cl.user_session.set("is_graph_rag_initialized", True)

        settings = await cl.ChatSettings(GRAPH_RAG_SETTINGS).send()

        llm, graph_workflow, graph_visualizer_tool = configure_graph_rag(
            llm_name=settings["llm_model"],
            neo4j_graph=cl.user_session.get("neo4j_graph"),
            embedder_model=cl.user_session.get("embedder_model"),
        )

        cl.user_session.set("llm", llm)
        cl.user_session.set("graph_workflow", graph_workflow)
        cl.user_session.set("graph_visualizer_tool", graph_visualizer_tool)
    elif chat_profile == "TAG":
        pass
        # Seingetku, hasil akhirnya harus di-chain ke StrOutputParser()
        ########################## CONTOH #############################
        # runnable = prompt | llm | StrOutputParser()
        # cl.user_session.set("runnable", runnable)
        ###############################################################


@cl.on_settings_update
async def setup_agent(settings):
    """
    TODO: Docstring
    """
    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "Graph-RAG":
        llm, graph_workflow, graph_visualizer_tool = configure_graph_rag(
            llm_name=settings["llm_model"],
            neo4j_graph=cl.user_session.get("neo4j_graph"),
            embedder_model=cl.user_session.get("embedder_model"),
        )

        cl.user_session.set("llm", llm)
        cl.user_session.set("graph_workflow", graph_workflow)
        cl.user_session.set("graph_visualizer_tool", graph_visualizer_tool)

    elif chat_profile == "TAG":
        pass


@cl.on_message
async def on_message(input_msg: cl.Message):
    """
    TODO: Docstring
    """
    chat_profile = cl.user_session.get("chat_profile")
    config = {"configurable": {"thread_id": cl.context.session.id}}

    if chat_profile == "Graph-RAG":
        graph_workflow = cl.user_session.get("graph_workflow")
        graph_visualizer_tool = cl.user_session.get("graph_visualizer_tool")

        await graph_rag_on_message(
            workflow=graph_workflow,
            graph_visualizer_tool=graph_visualizer_tool,
            input_msg=input_msg,
            config=config,
        )

    elif chat_profile == "TAG":
        pass
        ######################### CONTOH #############################
        # runnable = cl.user_session.get("runnable")  # type: Runnable
        # msg = cl.Message(content="")
        #
        # for chunk in await cl.make_async(runnable.stream)(
        #     {"question": message.content},
        #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        # ):
        #     await msg.stream_token(chunk)
        #
        # await msg.send()
        ##############################################################


if __name__ == "__main__":
    load_dotenv(".env")
