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
    "query": "Find 2 unique events in Northern California from 2016 with magnitude > 5.0. Get waveforms and plot them.",
    # "query": "Get waveforms and plot them for event between 2025-10-10 12:47:00 and  2025-10-10 12:49:00 near latitude 35.921, longitude -87.658. Download data from all stations and channels within 3 degrees.",
    # "query": "Find thee largest Earthquake to occur in Japan after 2009. Get Teleseismic distance waveforms and plot them.",
    "output_dir": "./temp"
}

agent = TremorsAgent(llm=llm)
result = agent._action.invoke(inputs)
