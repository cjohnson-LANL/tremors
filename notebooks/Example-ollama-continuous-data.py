import os
import sys
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from tremors import TremorsAgent
llm = ChatOpenAI(
    base_url="http://localhost:11438/v1",  
    api_key="ollama",
    model="gpt-oss:20b",
    temperature=0.7,
)

inputs = {
 "query": "Get continuous waveforms between latitudes 33 and 34, and longitudes -116 and -117 for February 1st to February 3rd, 2016. Look for BH* channels on the CI network. Do not use directory dates, use directory stats, and download response.",
    "output_dir": "./temp"
}

agent = TremorsAgent(llm=llm)
result = agent._action.invoke(inputs)