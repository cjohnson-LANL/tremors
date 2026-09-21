import os
from langchain_anthropic import ChatAnthropic
from tremors import TremorsAgent, approve_all

# Custom Anthropic-compatible endpoint (internal gateway, LiteLLM proxy, ...).
# Point base_url at the endpoint ROOT (the client appends /v1/messages) and
# pass your key via api_key. Leave base_url unset to hit the public API.
# Prefer environment variables so the key never lands in source:
#   export ANTHROPIC_BASE_URL=https://your-gateway.example.com
#   export ANTHROPIC_API_KEY=sk-ant-...
#
# NOTE: temperature is deliberately NOT set. Some gateway-hosted models reject the
# parameter outright with a 400 ("temperature is not supported"), so cli.py only
# forwards it for the anthropic backend when you pass --temperature explicitly.
# Add temperature=... here only if you know your model accepts it.
llm = ChatAnthropic(
    base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://your-gateway.example.com"),
    api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-ant-REPLACE_ME"),
    model="claude-opus-4-8",
    max_tokens=4096,
)

query = "Find 2 unique events in Northern California from 2016 with magnitude > 5.0. Get waveforms and plot them."
# query = "Get waveforms and plot them for event between 2025-10-10 12:47:00 and  2025-10-10 12:49:00 near latitude 35.921, longitude -87.658. Download data from all stations and channels within 3 degrees."
# query = "Find the largest Earthquake to occur in Japan after 2009. Get Teleseismic distance waveforms and plot them."

agent = TremorsAgent(llm=llm, output_dir="./temp")

# The agent pauses for approval before it retrieves anything. approve_all answers
# those gates inline so this script runs end to end. To review each request
# instead, drop on_interrupt: run() then returns status="Awaiting Input" with the
# pending request in result["interrupt"], and agent.resume(value) continues.
# TremorsAgent(..., interrupts=False) disables the gates entirely.
result = agent.run(query, on_interrupt=approve_all)
