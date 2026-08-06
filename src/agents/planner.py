# agents/planner.py

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from tools.planner_tools import context_tool, gap_filler_tool
from dotenv import load_dotenv

load_dotenv()

# to use Gemini:
from langchain.chat_models import init_chat_model
# To use local LLM
# from model.local_llm import ChatLlamaCpp


def plan(user_payload: dict):
    """
    Expected input:
        user_payload = {
            "sequence": "ATGG---TTCA",
            "gap_length": 870,
            "meta": {...}  # optional, any JSON-serializable metadata
        }
    This function builds a tool-calling agent that MUST:
      1) call context_tool first to retrieve context from the RAG corpus
      2) call gap_filler_tool next, passing EXACTLY the provided gap_length
    """

    # Basic validation
    if not isinstance(user_payload, dict):
        raise TypeError("plan(user_payload) expects a dict with 'sequence' and 'gap_length'.")
    if "sequence" not in user_payload:
        raise ValueError("Missing 'sequence' in user_payload.")
    if "gap_length" not in user_payload:
        raise ValueError("Missing 'gap_length' in user_payload.")

    sequence = user_payload.get("sequence")
    gap_length = user_payload.get("gap_length")
    meta = user_payload.get("meta", {})

    # System prompt guiding the tool order and usage
    system_prompt = """
You are a helpful bioinformatics agent that solves DNA sequence problems using tools.
You will be asked to fill in gaps in genomic sequences, where gaps are marked with "---".
First you must retrieve genomic context, then you must fill the gaps.

You MUST use these tools in the correct order:
1) context_tool
2) gap_filler_tool

CRITICAL RULES:
- The user provides an integer `gap_length` that represents the intended gap size.
- You MUST pass EXACTLY that `gap_length` to gap_filler_tool (do not recompute it).
- Use context_tool first to retrieve up to three best matching contexts.
- Then call gap_filler_tool with:
    sequence=<the sequence containing '---'>,
    rag_matches=<the contexts you just retrieved>,
    gap_length=<the exact integer provided by the user>.
- Be concise in your final answer.
"""

    # Prompt template: explicitly expose the fields to the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template(
                "sequence: {sequence}\n"
                "gap_length: {gap_length}\n"
                "meta: {meta}"
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Initialize the model
    # Option A: Remote model (Gemini via LangChain init_chat_model):
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    # Option B: Local LLM wrapper 
    # model = ChatLlamaCpp(base_url="http://localhost:8000")

    # Tools and agent wiring
    planner_tools = [context_tool, gap_filler_tool]
    agent = create_tool_calling_agent(llm=model, tools=planner_tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=planner_tools, verbose=True)

    # Invoke the agent with structured inputs matching the prompt variables
    return agent_executor.invoke(
        {
            "sequence": sequence,
            "gap_length": gap_length,
            "meta": meta,
        }
    )
