#!/usr/bin/env python3

import sys
import json
sys.path.append('src')
from src.progetto_crew_flows.tools.rag_tool import store_in_vectordb

# Test with the exact format that CrewAI is sending
test_content = {
    'content': '[{"topic": "football - premier league", "subject": "football", "content": "Test content for Premier League"}]'
}

print('=== TEST FIXED STORE_IN_VECTORDB ===')
print('Test input format:', type(test_content))
print('Test input keys:', list(test_content.keys()))

result = store_in_vectordb(**test_content)
print('Result:', result)
