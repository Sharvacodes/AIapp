import os
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
import streamlit as st
from langchain.schema import SystemMessage
from fastapi import FastAPI


load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

#Tool for search

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q":query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers = headers, data = payload)
    print(response.text)
    return response.text

search("Give me details about Chennai's weather today")

#2. Tool for scraping
def scrape_website(objective: str, url:str):

    print("Scraping website...")

    headers = {
        'Cache - control' : 'no-cache',
        'Content-Type':'application/json'
    }
    # Define the data to be sent in the request
    data = {"url" : url
    }

    #Convert Python object to JSON string
    data_json = json.dumps(data)

    #Send the post request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data = data_json)

    #Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTNENT: ", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

def summary(objective, content):
    llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n"], chunk_size = 10000, chunk_overlap = 500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """

    map_prompt_template = PromptTemplate(
        template = map_prompt, input_variables = ["text", "objective"])

    summary_chain = load_summarize_chain(
        llm = llm,
        chain_type = 'map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = True
    )

    output = summary_chain.run(input_documents = docs, objective = objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape website"""

    objective: str = Field(description="The obj and task that the users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any URL"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

#3. Create langchain agent with the tools above

tools = [Tool( 
            name = "Search",
            func = search,
            description =   "useful for when you need to answer questions about current events. You should ask targetted questions"),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content = """You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
    you do not make things up, you will try as hard as possible to gather facts & data to back up the research

    Please make sure, you complete the objective above with the following rules:
    1/ You should do enough research to gather as much as information as possible about the objective 
    2/ If there are URL of relevant links and articles, you will scrape it to gather more information 
    3/ After scraping and searching, you should think " Are there any new things I should search and scrape, based on the data I collected to increase the research quality? If answer is yes, continue; But, din't do this more than 3 iterations 
    4/ You should not make things up, you should only write facts & data you that you have gathered. 
    5/ In the final output, you should include all reference data & links to back up your research
    6/ In the final output, you should include all reference data & links to back up your research
    """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name = "memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(memory_key = "memory", return_messages = True, llm = llm, max_token_limit = 1000)

agent = initialize_agent(
    tools,
    llm,
    agent = AgentType.OPENAI_FUNCTIONS,
    verbose = True,
    agent_kwargs = agent_kwargs,
    memory = memory,
)   

#4. Use streamlit to create a webapp

# def main():
#     st.set_page_config(page_title = "AI research agent", page_icon = ":bird")
#     st.header("AI research agent :bird:")
#     query = st.text_input("Research goal")

#     if query:
#         st.write("Doing research for ", query)
#         result = agent({"input":query})

#         st.info(result['output'])

# if __name__ == '__main__':

#     #question = "india world cup"
#     #print(" Results :" + agent({"input":question}))
#     main()

#5. Set this as an API endpoint via FastAPI 

app = FastAPI()

class Query(BaseModel):
    query: str 

@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return content 
