"""LLM text generation evaluation workflow"""

import copy
import time
import uuid
from typing import List
from tqdm import tqdm
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from ragas import EvaluationDataset
from ..agent import create_agent

# from langgraph.graph import MessagesState
# from langgraph.prebuilt.chat_agent_executor import _get_prompt_runnable


HYBRID_CYPHER_CONTEXT_TEMPLATE = """
## **Daftar Pasal Peraturan Perundang-Undangan yang (Mungkin) Relevan untuk Menjawab Kueri:**
--------------------------------------------------------------------------------

{context}
""".strip()


TEXT2CYPHER_CONTEXT_TEMPLATE = """
### **Hasil Pembuatan Kode Cypher:**
```{cypher}```

### **Hasil Eksekusi Kode Cypher ke Database:**
{context}
""".strip()


def run_text_generation_workflow(
    evaluation_dataset: EvaluationDataset,
    experiment_name: str,
    *,
    expected_tool_call_names: List[str],
    generated_cypher_results: List[str],
    llm: BaseChatModel,
    verbose: bool = True,
) -> EvaluationDataset:
    """
    TODO: Docstring
    """
    evaluation_dataset = copy.deepcopy(evaluation_dataset)
    agent = create_agent(model=llm, tools=[])
    # counter = 0

    for data, tool_name, cypher in tqdm(
        iterable=list(
            zip(evaluation_dataset, expected_tool_call_names, generated_cypher_results)
        ),
        desc=f"Running llm_text_generation: `{experiment_name}`",
        disable=not verbose,
    ):
        # if counter % 15 == 0 and counter != 0:
        #     time.sleep(60)
      
        tool_call_id = f"run-{uuid.uuid4()}-0"

        # Formatting ToolMessage content
        if "text2cypher" in tool_name and cypher:
            tool_message_content = TEXT2CYPHER_CONTEXT_TEMPLATE.format(
                cypher=cypher, context="[" + " ".join(data.retrieved_contexts) + "]"
            )
        elif "hybrid" in tool_name:
            tool_message_content = HYBRID_CYPHER_CONTEXT_TEMPLATE.format(
                context="\n\n".join(data.retrieved_contexts)
            )
        else:
            print("Unknown `tool_name`, skipping `user_input`: " f"{data.user_input}")
            continue

        # Create fake "messages" state history
        state = {
            "messages": [
                HumanMessage(content=data.user_input),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": {"query": data.user_input},
                            "id": tool_call_id,
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=tool_message_content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                ),
            ]
        }

        response = agent.invoke(state)
        data.response = response["messages"][-1].content

        # counter += 1

    return evaluation_dataset
