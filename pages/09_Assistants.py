import streamlit as st
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
import openai as client
import json
import os
import time


if "api_key" not in st.session_state:
    st.session_state["api_key"] = None


# llm = ChatOpenAI(temperature=0.1, api_key=st.session_state["api_key"])


def get_wikipedia(inputs):
    wiki = WikipediaAPIWrapper()

    query = inputs["query"]

    return wiki.run(f"research of  {query}")


def get_duckduckgo(inputs):
    query = inputs["query"]
    # ddg =  DuckDuckGoSearchResults()
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(f"research of  {query}")


functions_map = {
    "get_wikipedia": get_wikipedia,
    "get_duckduckgo": get_duckduckgo,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_wikipedia",
            "description": "Given the query return the results of searching the Wikipedia site",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "query to search for",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_duckduckgo",
            "description": "Given the query return the results of searching the DuckDuckGoSearchTo site",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "query to search for",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    print("*****", run.required_action.submit_tool_outputs.tool_calls, "\n\n")
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outpus = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outpus
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)

    messages = list(messages)
    print(messages)
    for message in messages:
        print(f"{message.role}:{message.content[0].text.value}")

    print("===========================================")
    messages.reverse()
    print(messages)
    for message in messages:
        print(f"{message.role}:{message.content[0].text.value}")


if "assistant_id" not in st.session_state:
    assistant = client.beta.assistants.create(
        name="Search Assistant",
        instructions="You help users do research information. It takes a query as an argument.",
        model="gpt-4-1106-preview",
        tools=functions,
    )
    st.session_state["assistant_id"] = assistant.id


@st.cache_data(show_spinner="Loading website...")
def load_website(api_key):
    if st.session_state["api_key"]:
        try:

            assistant_id = st.session_state["assistant_id"]
            print(assistant_id)
            thread = client.beta.threads.create(
                messages=[{"role": "user", "content": "Research about the xz backdoor"}]
            )
            run = client.beta.threads.runs.create(
                thread_id=thread.id, assistant_id=assistant_id
            )
            result = get_run(run.id, thread.id).status
            print(result)

            time.sleep(2)
            print(get_run(run.id, thread.id).status)
            submit_tool_outputs(run.id, thread.id)

        except Exception as e:
            st.error(f"{e} - Please Check API_KEY ")
            st.session_state["api_key"] = None
            return None
    return result


st.set_page_config(page_title="Assistant", page_icon="🧑‍⚕️")


st.markdown(
    """
    # Assistant
            
    Given the query return the results of searching the Wikipedia & DuckDuckgo site
"""
)


with st.sidebar:
    api_key = st.text_input(":blue[OpenAI API_KEY]", type="password")


if api_key:
    st.session_state["api_key"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    retriever = load_website(api_key)


# get_messages(thread.id)

# get_run(run.id, thread.id).status
# submit_tool_outputs(run.id, thread.id)
