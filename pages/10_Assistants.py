import streamlit as st
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from openai import OpenAI
import json
import os
import time


###
with st.sidebar:
    if "api_key" not in st.session_state:
        st.session_state["api_key]"] = ""

    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.markdown(
        """
        Github 링크: https://github.com/moonssa/gpt/commit/747ccd1bcf09c33c79d26a9091a56c944ff503e7
        """
    )

client = OpenAI(api_key=api_key)


def get_wikipedia(inputs):
    wiki = WikipediaAPIWrapper()
    query = inputs["query"]
    # return wiki.run(f"research of  {query}")
    return wiki.run(query)


def get_duckduckgo(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchAPIWrapper()
    # return ddg.run(f"research of  {query}")
    return ddg.run(query)


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
                        "description": "query to search for. Example query: Research about the XZ backdoor",
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
                        "description": "query to search for. Example query: Research about the XZ backdoor",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


@st.cache_data
def create_assistant():
    return client.beta.assistants.create(
        name="Search Assistant",
        # instructions="You are the user's research assistant.Search for query.If there is a website url list in the search results list, extract each website content as a text",
        instructions="You are the user's research assistant.Search for query.",
        model="gpt-4-1106-preview",
        tools=functions,
    )


@st.cache_data
def create_thread(content):
    return client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ]
    )


@st.cache_data
def create_run(assistant_id, thread_id):
    return client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )


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
        st.write(f"{message.role}: {message.content[0].text.value}")
        print(f"{message.role}:{message.content[0].text.value}")


def get_wikipedia(query):
    wiki = WikipediaAPIWrapper()
    return wiki.run(f"research of {query}")


def get_duckduckgo(query):
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(f"research of {query}")


user_query = st.text_input("Enter your search query:")

if st.button("Search"):
    if not user_query:
        st.warning("Please enter a query to search.")
    else:
        try:
            assistant = create_assistant()
            print(assistant)
            thread = create_thread(user_query)
            print(thread)
            run = create_run(assistant.id, thread.id)

            while True:
                run = client.beta.threads.runs.poll(
                    run.id,
                    thread.id,
                    poll_interval_ms=500,
                    timeout=20,
                )
                print(run.status)
                if run.status == "requires_action":
                    submit_tool_outputs(run.id, thread.id)
                    break
                if run.status == "completed":
                    break
                if run.status in ("expired", "failed"):
                    st.write(run.status)

            get_messages(thread.id)
        except Exception as e:
            st.error(f"{e}")
