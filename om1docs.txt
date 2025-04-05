OpenMind home pagedark logo

    Support
    Sign Up

Home
Guides
API Reference
Research
API Reference

    Introduction
    Authentication
    API Price

Core Endpoints

    POST
    LLM Chat Completions
    POST
    ElevenLabs Text to Speech
    POST
    Riva Text to Speech

Core Websocket Endpoints

    VILA VLM
    Google Speech Recognition
    Riva Speech Recognition

Core Endpoints
LLM Chat Completions

Get completions for a chat message
POST
/
{provider}
/
chat
/
completions

OpenMind integrates with multiple language models (LLMs) to provide chat completions. This endpoint allows you to interact with different LLM providers to generate chat completions.
Authorizations
​
x-api-key
string
header
required
Path Parameters
​
provider
enum<string>
required

The provider of the LLM (openai, deepseek, gemini)
Available options: openai, 
deepseek, 
gemini, 
xai 
Body
application/json
​
model
enum<string>
required

The model to use for completion.
Available options: gpt-4o, 
gpt-4o-mini, 
deepseek-chat, 
gemini-2.0-flash-exp, 
grok-2-latest 
​
messages
object[]
required
Response
200
application/json
Successful response
​
choices
object[]
required

A list of generated response choices.
​
created
integer
required

The Unix timestamp of response creation.
Example:

1742147458
​
id
string
required

The unique identifier of the completion request.
Example:

"chatcmpl"
​
model
string
required

The model used for completion.
Example:

"gpt-4o-2024-08-06"
​
object
string
required

The object type of the response.
Example:

"chat.completion"
​
usage
object
required

Token usage details.
​
service_tier
string

The service tier used for the request.
Example:

"default"
​
system_fingerprint
string

A fingerprint identifier for system consistency.
Example:

"fingerprint"

Was this page helpful?
API Price
ElevenLabs Text to Speech
x
github
Powered by Mintlify
Copy

import requests

url = "https://api.openmind.org/api/core/{provider}/chat/completions"

payload = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "system",
            "content": "<string>"
        }
    ]
}
headers = {
    "x-api-key": "<api-key>",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)

Copy

{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "annotations": [],
        "content": "Hello! How can I assist you today?",
        "refusal": null,
        "role": "assistant"
      }
    }
  ],
  "created": 1742147458,
  "id": "chatcmpl",
  "model": "gpt-4o-2024-08-06",
  "object": "chat.completion",
  "service_tier": "default",
  "system_fingerprint": "fingerprint",
  "usage": {
    "completion_tokens": 10,
    "completion_tokens_details": {
      "accepted_prediction_tokens": 0,
      "audio_tokens": 0,
      "reasoning_tokens": 0,
      "rejected_prediction_tokens": 0
    },
    "prompt_tokens": 8,
    "prompt_tokens_details": {
      "audio_tokens": 0,
      "cached_tokens": 0
    },
    "total_tokens": 18
  }
}

LLM Chat Completions - OpenMind