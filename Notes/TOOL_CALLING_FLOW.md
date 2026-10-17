# How an LLM Tool Call Works: From API Request to Tool Execution

A custom tool call is usually a two-request loop. The LLM produces a structured request to call a tool; your application receives that request, runs the real function, and sends its result back to the LLM.

```text
Your app                    OpenAI API / LLM                  Your tool
   |                               |                              |
   |-- POST prompt + tool schema ->|                              |
   |                               |-- model generates call ------|
   |<-- function_call object ------|                              |
   |                                                              |
   |-- validate arguments --------------------------------------->|
   |<------------------------- result -----------------------------|
   |                                                              |
   |-- POST function_call_output ->|                              |
   |                               |-- model reads tool result     |
   |<-- final natural-language ----|                              |
```

## 1. Your application defines the tool

Suppose your application has this real function:

```python
def get_weather(location: str, unit: str) -> dict:
    # This could call a database or external weather API.
    return {
        "location": location,
        "temperature": 31,
        "unit": unit,
        "condition": "cloudy",
    }
```

The LLM cannot see or execute this Python code. You only send it a description of the function:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get the current weather for a location.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City and country, for example Chennai, India"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location", "unit"],
    "additionalProperties": false
  },
  "strict": true
}
```

This schema tells the model:

- The tool's name
- When it should be used
- Which arguments it accepts
- Which values and types are valid

OpenAI documents that these schemas are inserted into the model context in a tool-specific syntax and count as input tokens. With `strict: true`, generation is constrained to follow the supplied JSON schema. See the [OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling).

## 2. Your application sends the first API request

Conceptually, the raw HTTP request is:

```http
POST https://api.openai.com/v1/responses
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

```json
{
  "model": "gpt-5.4",
  "input": "What is the weather in Chennai?",
  "tools": [
    {
      "type": "function",
      "name": "get_weather",
      "description": "Get the current weather for a location.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location", "unit"],
        "additionalProperties": false
      },
      "strict": true
    }
  ],
  "tool_choice": "auto"
}
```

`tool_choice: "auto"` means the model can:

- Answer without a tool
- Call one tool
- Call multiple tools

You can also use `"required"` or force a particular function.

## 3. What happens inside the model service

The API validates the request and constructs the model's context from:

```text
System/developer instructions
+ user message
+ available tool names
+ tool descriptions
+ argument schemas
```

During generation, the model effectively decides between outputs such as:

```text
Generate normal assistant text
```

or:

```text
Generate a function call
```

For this question, it recognizes that it does not possess live weather data and selects `get_weather`.

It then generates:

- The function name: `get_weather`
- The arguments: `{"location":"Chennai, India","unit":"celsius"}`

This is not ordinary text that your code must extract from a sentence. The API exposes it as a typed `function_call` output item.

The exact internal neural computation is proprietary, but at the API boundary the model is generating tool-call tokens that the service parses into a structured output item. Strict mode constrains the argument generation to the schema.

## 4. The first API response contains a request, not a weather result

The response looks approximately like this:

```json
{
  "id": "resp_abc123",
  "status": "completed",
  "output": [
    {
      "type": "function_call",
      "id": "fc_123",
      "call_id": "call_weather_123",
      "name": "get_weather",
      "arguments": "{\"location\":\"Chennai, India\",\"unit\":\"celsius\"}",
      "status": "completed"
    }
  ]
}
```

Notice that `arguments` is a JSON-encoded string. Your application must parse it:

```python
arguments = json.loads(item.arguments)
```

Most importantly, the tool has still not run.

The response means:

> Application, please execute `get_weather` with these arguments and return the result associated with `call_weather_123`.

For custom functions, OpenAI does not have your Python process, credentials, database connection, or function implementation. Therefore, your application must execute it.

## 5. Your application dispatches the call

A basic dispatcher looks like this:

```python
import json

registered_tools = {
    "get_weather": get_weather,
}

for item in response.output:
    if item.type != "function_call":
        continue

    tool = registered_tools.get(item.name)

    if tool is None:
        result = {"error": f"Unknown tool: {item.name}"}
    else:
        arguments = json.loads(item.arguments)
        result = tool(**arguments)
```

The actual invocation is this ordinary Python operation:

```python
result = get_weather(
    location="Chennai, India",
    unit="celsius",
)
```

It produces:

```json
{
  "location": "Chennai, India",
  "temperature": 31,
  "unit": "celsius",
  "condition": "cloudy"
}
```

A production dispatcher should also:

- Allow only registered tool names
- Validate arguments again
- Check user authorization
- Apply timeouts
- Catch exceptions
- Require approval for sensitive operations
- Avoid blindly trusting model-generated arguments

The model proposes an action; your application remains the authority that decides whether to execute it.

## 6. Your application sends the tool result back

Now your application makes a second API request:

```http
POST https://api.openai.com/v1/responses
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

```json
{
  "model": "gpt-5.4",
  "previous_response_id": "resp_abc123",
  "input": [
    {
      "type": "function_call_output",
      "call_id": "call_weather_123",
      "output": "{\"location\":\"Chennai, India\",\"temperature\":31,\"unit\":\"celsius\",\"condition\":\"cloudy\"}"
    }
  ]
}
```

Two identifiers serve different purposes:

- `previous_response_id` reconnects this request to the previous model response.
- `call_id` connects this particular result to the corresponding function call.

The `output` is normally a string. It can contain serialized JSON, plain text, an error, or a success message.

## 7. The LLM processes the tool result

The model now receives context equivalent to:

```text
User: What is the weather in Chennai?

Assistant requested:
get_weather(location="Chennai, India", unit="celsius")

Tool returned:
{
  "temperature": 31,
  "condition": "cloudy"
}
```

It then generates a normal assistant message:

```json
{
  "id": "resp_def456",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "It is currently 31°C and cloudy in Chennai."
        }
      ]
    }
  ]
}
```

The documented flow is: send tools, receive a call, execute application code, return its output, and receive a final response or another tool call. See the [OpenAI tool-calling flow](https://developers.openai.com/api/docs/guides/function-calling#the-tool-calling-flow).

## Complete Python loop

```python
import json
from openai import OpenAI

client = OpenAI()


def get_weather(location: str, unit: str) -> dict:
    return {
        "location": location,
        "temperature": 31,
        "unit": unit,
        "condition": "cloudy",
    }


registered_tools = {
    "get_weather": get_weather,
}

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

response = client.responses.create(
    model="gpt-5.4",
    input="What is the weather in Chennai?",
    tools=tools,
)

while True:
    calls = [
        item
        for item in response.output
        if item.type == "function_call"
    ]

    if not calls:
        print(response.output_text)
        break

    tool_outputs = []

    for call in calls:
        try:
            arguments = json.loads(call.arguments)
            function = registered_tools[call.name]
            result = function(**arguments)
            output = json.dumps(result)
        except Exception as error:
            output = json.dumps({"error": str(error)})

        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            }
        )

    response = client.responses.create(
        model="gpt-5.4",
        previous_response_id=response.id,
        input=tool_outputs,
        tools=tools,
    )
```

The `while` loop is important because the next model response might contain another tool call instead of final text. Agent frameworks and the Agents SDK mainly automate this orchestration loop, registration, tracing, validation, and error handling.

## Custom, built-in, and MCP tools

The execution location depends on the type of tool:

| Tool type | Who executes it? |
| --- | --- |
| Custom function | Your application or backend |
| OpenAI built-in tool | The OpenAI platform |
| MCP tool | An MCP server, reached through the surrounding runtime or platform |

In every case, the model selects and parameterizes the tool. The surrounding runtime performs or coordinates the actual operation.

## Core takeaway

The LLM never directly reaches into your application and calls a Python function. It generates a typed request containing a tool name and arguments. Your runtime validates that request, maps the name to registered code, runs it, and returns the result to the model for the next inference step.
