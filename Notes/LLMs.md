# LLM tools, workflows, knowledge retries and many more

The LLM only decides:

1. Which tool should be called.
2. What arguments should be passed.
3. How to use the result afterward.

Our application is the **orchestrator** actually executes the program.

## The usual architecture

```text
User
  │
  ▼
Your application / agent runtime
  │
  ├── Sends prompt + available tool definitions to LLM
  │
  ▼
LLM
  │
  ├── Returns normal text
  │        or
  └── Returns structured tool call:
          {
            "tool": "get_weather",
            "arguments": {
              "city": "Hyderabad"
            }
          }
  │
  ▼
Your application validates the call
  │
  ├── Runs Python/Rust function locally
  ├── Calls an internal service
  └── Calls an external API
  │
  ▼
Tool result is sent back to the LLM
  │
  ▼
LLM produces the final answer
```

The important component is not another LLM. It is a relatively small **tool execution loop** around the primary LLM.

## The LLM is not expected to understand your runtime

Suppose you have this Python function:

```python
def get_order_status(order_id: str) -> dict:
    ...
```

You expose a description of it to the LLM:

```python
TOOLS = [
    {
        "name": "get_order_status",
        "description": (
            "Retrieve the current status of a customer order. "
            "Use this when the user asks whether an order has shipped, "
            "is delayed, or has been delivered."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The customer's order identifier."
                }
            },
            "required": ["order_id"],
            "additionalProperties": False
        }
    }
]
```

The model does not receive or run the implementation. It receives the **interface contract**.

It might return:

```json
{
  "name": "get_order_status",
  "arguments": {
    "order_id": "ORD-1042"
  }
}
```

Your code then maps that tool name to the actual function:

```python
TOOL_HANDLERS = {
    "get_order_status": get_order_status,
}
```

And executes it:

```python
handler = TOOL_HANDLERS[tool_call["name"]]
result = handler(**tool_call["arguments"])
```

## General wording does not require keyword matching

This concern:

> If we statically type tools into an array, general user wording may break.

That would be true if your application were doing something like:

```python
if "order status" in user_message:
    get_order_status(...)
```

But that is not how model-based tool calling normally works.

The model uses the tool's:

- Name
- Description
- Parameter descriptions
- Current conversation
- Examples, when provided

So all these could select the same tool:

```text
Where is order ORD-1042?
Has my package shipped?
Is ORD-1042 delayed?
When will that order arrive?
Can you check what happened to my recent purchase?
```

The model performs the semantic mapping from the user's natural language to the structured tool call.

Your program remains deterministic after the tool call has been selected.

## A minimal orchestration loop

Conceptually, the implementation looks like this:

```python
messages = [
    {"role": "user", "content": user_message}
]

for _ in range(MAX_TOOL_ROUNDS):
    response = call_llm(
        messages=messages,
        tools=TOOL_SCHEMAS,
    )

    if not response.tool_calls:
        return response.text

    messages.append(response.assistant_message)

    for call in response.tool_calls:
        tool_name = call.name
        arguments = validate_arguments(tool_name, call.arguments)

        try:
            result = execute_tool(tool_name, arguments)
        except Exception as exc:
            result = {
                "ok": False,
                "error": safe_error_message(exc),
            }

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": serialize(result),
        })

raise RuntimeError("Maximum tool-call rounds exceeded")
```

Different LLM providers represent the messages differently, but the architecture is largely the same.

## You do not necessarily need HTTP

A cloud function or HTTP server is only one deployment option.

### Same-process execution

For Python tools:

```python
result = python_function(**arguments)
```

This normally has negligible dispatch latency.

### Local subprocess

For a Rust binary or isolated Python script:

```python
subprocess.run(
    ["./my_rust_tool", "--input", value],
    capture_output=True,
    timeout=10,
)
```

This adds process-startup overhead but not network latency.

### Foreign-function interface

You can expose Rust code to Python through mechanisms such as:

- PyO3
- C-compatible FFI
- Shared libraries
- Unix sockets

This can provide very low latency.

### Local service

Run the tool as a service on the same machine:

```text
LLM orchestrator → localhost service → result
```

This gives process isolation while avoiding external network latency.

### Remote HTTP service

Use HTTP when the tool:

- Needs independent scaling
- Uses separate credentials
- Must be shared by multiple applications
- Runs in another security boundary
- Needs a different runtime or machine type

HTTP should be a deployment decision, not a requirement of tool calling.

## Where latency really comes from

The most expensive sequence is usually:

```text
LLM request
→ tool execution
→ second LLM request
→ second tool execution
→ third LLM request
→ final answer
```

The problem is often not the Python or Rust tool. It is the repeated model round trips.

You can reduce this with several techniques.

### Parallel tool calls

When tools are independent:

```text
get_customer
get_open_invoices
get_support_tickets
```

Execute all three simultaneously rather than serially.

```python
results = await asyncio.gather(
    execute_tool(call_1),
    execute_tool(call_2),
    execute_tool(call_3),
)
```

### Composite tools

Instead of exposing:

```text
get_customer
get_orders
get_payments
get_shipments
```

You could expose:

```text
get_customer_account_summary
```

Internally, that tool calls all four systems in parallel.

This reduces model round trips, but it makes the tool less reusable. There is a trade-off between atomic and composite tools.

### Return small results

Do not return a 10 MB database response to the model.

Return:

```json
{
  "status": "shipped",
  "carrier": "DHL",
  "estimated_delivery": "2026-07-16"
}
```

Instead of every internal order record and audit event.

Large tool outputs increase:

- Input tokens
- Latency
- Cost
- Risk of confusing the model

### Let tools perform deterministic processing

Bad design:

```text
Tool returns 50,000 database rows.
LLM calculates the total.
```

Better design:

```text
Tool performs the SQL aggregation.
Tool returns the total and relevant breakdown.
LLM explains it.
```

Use code for deterministic computation and the LLM for interpretation.

## Tool definitions do contribute to token usage

You are correct that tools can increase LLM cost, but the exact reason matters.

Costs can increase because:

1. Tool definitions are included in the model context.
2. Tool results are sent back into the conversation.
3. Every additional model round trip consumes tokens.
4. Large descriptions and schemas use more input tokens.

The actual execution of a local Python function does not inherently consume LLM tokens.

For example:

```text
Calling calculate_tax() locally:
  No LLM token cost

Sending 20 KB of tool output back to the model:
  LLM input token cost
```

## Avoid sending hundreds of tools every time

When an agent has a large tool catalog, sending all tools to the model is inefficient and reduces tool-selection accuracy.

A practical architecture is:

```text
User request
    │
    ▼
Tool catalog retrieval
    │
    ├── Select 5–15 possibly relevant tools
    ▼
Main LLM receives only those tools
```

The catalog selection does not necessarily require a second LLM.

You can use:

- Embedding similarity
- Namespaces
- Product or page context
- User permissions
- Deterministic category filters
- A small classifier
- A smaller model, only when needed

For example:

```text
User: "Why did my deployment fail?"

Candidate tool namespace:
- deployments.get_status
- deployments.get_logs
- deployments.get_configuration
- repositories.get_commit

Excluded:
- accounting.*
- calendar.*
- customer_support.*
```

This is sometimes called **dynamic tool retrieval** or **tool routing**.

## A small local LLM is optional, not the default

Your proposed local lightweight LLM could perform:

- Tool routing
- Argument extraction
- Intent classification
- Filtering the tool catalog

But it introduces:

- Another model to operate
- More prompts
- More failure modes
- More observability requirements
- Potential differences between the router and main model
- Additional latency

For a small or medium collection of tools, the primary LLM should normally select the tool directly.

A small model becomes useful when:

- You have hundreds or thousands of tools.
- Most requests do not need an expensive model.
- Routing is simple and high volume.
- Data must remain local.
- You want to reject obviously irrelevant requests before calling the main model.

A common production pattern is:

```text
Deterministic filtering
        ↓
Optional cheap classifier
        ↓
Main LLM with a small relevant tool set
        ↓
Deterministic tool executor
```

Not:

```text
One LLM for every individual tool decision
```

## Recommended architecture

For your case, I would start with this:

```text
┌─────────────────────────────┐
│ User request                │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Agent/orchestrator          │
│                             │
│ - conversation state        │
│ - tool schemas              │
│ - validation                │
│ - permission checks         │
│ - execution loop            │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Primary LLM                 │
│ Selects tool + arguments    │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Tool registry               │
│                             │
│ name → implementation       │
│                             │
│ local Python function       │
│ Rust subprocess / FFI       │
│ local service               │
│ remote API                  │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Validated compact result    │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Primary LLM final response  │
└─────────────────────────────┘
```

Start with local function calls wherever possible. Introduce HTTP only for isolation, sharing, scaling, or security reasons. Introduce a smaller routing model only after the size of the tool catalog or request volume demonstrates that you need one.

The key distinction is:

> **The LLM chooses and describes an action. The orchestrator authorizes and executes it.**

The LLM should never be the trusted execution boundary. Your application must validate arguments, enforce permissions, apply timeouts, limit retries, sanitize outputs, and decide which tools the model is allowed to see.