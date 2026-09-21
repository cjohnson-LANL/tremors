import os
import sys
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from tremors import TremorsAgent, approve_all
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",  # matches the cli.py default
    api_key="ollama",
    model="gpt-oss:20b",
    temperature=0.7,
)

query = "Get continuous waveforms between latitudes 33 and 34, and longitudes -116 and -117 for February 1st to February 3rd, 2016. Look for BH* channels on the CI network. Do not use directory dates, use directory stats, and download response."

agent = TremorsAgent(llm=llm, output_dir="./temp")

# The agent pauses for approval before it retrieves anything. approve_all answers
# those gates inline so this script runs end to end. To review each request
# instead, drop on_interrupt: run() then returns status="Awaiting Input" with the
# pending request in result["interrupt"], and agent.resume(value) continues.
# TremorsAgent(..., interrupts=False) disables the gates entirely.
result = agent.run(query, on_interrupt=approve_all)