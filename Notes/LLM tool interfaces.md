# LLM Tool Interfaces

The “interface” between an LLM and a tool is usually **not a direct connection between the model and Python/Rust program**.

It is a structured messaging protocol:

```text
Tool definition → LLM requests a tool call → application executes it
→ application returns structured result → LLM continues
```

There are really **three separate contracts**.

## 1. Model-facing tool contract

This tells the LLM:

* What the tool does
* When it should be used
* What arguments it accepts
* Which arguments are required
* What values are allowed

For example:

```json
{
  "type": "function",
  "function": {
    "name": "get_order_status",
    "description": "Get the current shipping and delivery status of an order.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "description": "The unique order identifier, such as ORD-12345."
        },
        "include_history": {
          "type": "boolean",
          "description": "Whether to include previous order status changes.",
          "default": false
        }
      },
      "required": ["order_id"],
      "additionalProperties": false
    }
  }
}
```

This is generally based on JSON Schema.

The model sees this contract, but it does not see or execute your Python implementation.

Your implementation might be:

```python
def get_order_status(
    order_id: str,
    include_history: bool = False,
) -> dict:
    ...
```

Or Rust:

```rust
fn get_order_status(
    order_id: String,
    include_history: bool,
) -> Result<OrderStatus, ToolError> {
    ...
}
```

The language does not matter to the model. The model only understands the exposed schema.

---

## 2. Execution contract

This is the contract between your orchestrator and the actual program.

Suppose the user says:

> Has my order ORD-12345 shipped yet?

The model might return a structured request similar to:

```json
{
  "id": "call_7a31",
  "type": "function",
  "function": {
    "name": "get_order_status",
    "arguments": {
      "order_id": "ORD-12345",
      "include_history": false
    }
  }
}
```

This is not the answer to the user. It is the model asking your application:

> Please execute this function with these arguments.

Your application receives that request and performs several steps.

```python
tool_call = response.tool_calls[0]

tool_name = tool_call.function.name
arguments = tool_call.function.arguments

validated_arguments = validate(
    tool_name=tool_name,
    arguments=arguments,
)

result = TOOL_REGISTRY[tool_name](**validated_arguments)
```

The registry might look like:

```python
TOOL_REGISTRY = {
    "get_order_status": get_order_status,
    "cancel_order": cancel_order,
    "get_customer": get_customer,
}
```

The model-generated name is therefore mapped to trusted application code.

The model cannot arbitrarily call:

```text
os.system
rm -rf
read_database_password
```

unless you explicitly expose such functions in the registry—which you should not.

## The important security boundary

Never execute a function directly using arbitrary model-generated names.

Bad:

```python
function = globals()[tool_name]
result = function(**arguments)
```

Better:

```python
TOOL_REGISTRY = {
    "get_order_status": get_order_status,
    "cancel_order": cancel_order,
}

if tool_name not in TOOL_REGISTRY:
    raise UnknownToolError(tool_name)
```

The registry is an allowlist.

---

## 3. Tool-result contract

After execution, your application sends the result back to the model.

For example, the Python program returns:

```json
{
  "order_id": "ORD-12345",
  "status": "shipped",
  "carrier": "DHL",
  "tracking_number": "DHL987654",
  "estimated_delivery": "2026-07-16"
}
```

Your orchestrator wraps it in a tool-result message:

```json
{
  "role": "tool",
  "tool_call_id": "call_7a31",
  "content": {
    "ok": true,
    "data": {
      "order_id": "ORD-12345",
      "status": "shipped",
      "carrier": "DHL",
      "estimated_delivery": "2026-07-16"
    }
  }
}
```

The `tool_call_id` connects the result to the model's earlier request.

The model then receives the conversation approximately like this:

```json
[
  {
    "role": "user",
    "content": "Has order ORD-12345 shipped yet?"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "id": "call_7a31",
        "name": "get_order_status",
        "arguments": {
          "order_id": "ORD-12345"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_7a31",
    "content": {
      "ok": true,
      "data": {
        "status": "shipped",
        "carrier": "DHL",
        "estimated_delivery": "2026-07-16"
      }
    }
  }
]
```

The model can now produce:

> Your order has shipped with DHL and is expected to arrive on July 16, 2026.

The model did not query the order system itself. It reasoned from the result your application supplied.

---

# The complete lifecycle

The process is usually:

```text
1. Application sends:
   - User message
   - Conversation history
   - Available tool definitions

2. Model responds with:
   - Normal text
   or
   - One or more tool calls

3. Application:
   - Checks that the tool is permitted
   - Parses the arguments
   - Validates them against the schema
   - Checks authorization
   - Executes the implementation
   - Captures errors and timeouts

4. Application sends the tool result back to the model.

5. Model:
   - Produces the final answer
   - Or requests another tool call
```

A model may make multiple rounds:

```text
User request
    ↓
get_customer_by_email
    ↓
Customer ID returned
    ↓
get_customer_orders
    ↓
Orders returned
    ↓
get_order_status
    ↓
Final answer
```

Your orchestrator should impose a maximum number of rounds to prevent loops.

```python
MAX_TOOL_ROUNDS = 5
```

---

# Tool input should be treated as untrusted

Even though the arguments come from the LLM, they must be treated like external user input.

For example, the model may generate:

```json
{
  "order_id": "'; DROP TABLE orders; --"
}
```

Or:

```json
{
  "limit": 1000000000
}
```

Or:

```json
{
  "file_path": "../../etc/passwd"
}
```

Therefore, the executor should perform normal validation.

```python
class GetOrderStatusInput(BaseModel):
    order_id: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^ORD-[0-9]+$",
    )
    include_history: bool = False
```

Validation:

```python
validated = GetOrderStatusInput.model_validate(arguments)
result = get_order_status(**validated.model_dump())
```

Schema validation answers:

> Is this structurally valid?

Authorization answers:

> Is this user allowed to perform this operation?

Those are different checks.

For example:

```python
if not current_user.can_view_order(validated.order_id):
    raise PermissionDenied()
```

The LLM should not be trusted to enforce permissions.

---

# A good tool contract is semantic, not implementation-specific

Consider a badly designed tool:

```json
{
  "name": "execute_query",
  "description": "Execute SQL.",
  "parameters": {
    "type": "object",
    "properties": {
      "sql": {
        "type": "string"
      }
    }
  }
}
```

This exposes too much power and requires the model to understand your database schema.

A better contract:

```json
{
  "name": "find_customer_orders",
  "description": "Find orders belonging to a customer.",
  "parameters": {
    "type": "object",
    "properties": {
      "customer_id": {
        "type": "string"
      },
      "status": {
        "type": "string",
        "enum": [
          "pending",
          "processing",
          "shipped",
          "delivered",
          "cancelled"
        ]
      },
      "limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
        "default": 10
      }
    },
    "required": ["customer_id"],
    "additionalProperties": false
  }
}
```

The model asks for a business operation. Your program decides how to perform it.

```text
LLM-facing capability:
    find_customer_orders

Internal implementation:
    SQL, REST, GraphQL, Redis, Rust service, or anything else
```

You can change the backend without changing the tool interface.

---

# Tool descriptions determine when tools are selected

The model does not merely use the function name. It relies heavily on the description.

Weak description:

```json
{
  "name": "search",
  "description": "Search for data."
}
```

The model cannot reliably determine:

* What data?
* When should it use this?
* Is it searching customers, documents, or the web?
* What should it not use it for?

Better:

```json
{
  "name": "search_internal_documents",
  "description": "Search the organization's internal policy and engineering documents. Use this for questions about internal processes, architecture, policies, and project documentation. Do not use it for public web information."
}
```

A useful tool description normally explains:

1. What the tool does
2. When to use it
3. When not to use it
4. Important limitations
5. Meaning of the returned data

Example:

```json
{
  "name": "get_account_balance",
  "description": "Return the current available and pending balances for one bank account. Use only when the user asks about a specific account's balance. This tool does not return transaction history and does not initiate transfers."
}
```

This reduces incorrect tool selection.

---

# Parameter descriptions also affect accuracy

This is not enough:

```json
{
  "start": {
    "type": "string"
  }
}
```

The model does not know whether `start` means:

* Start date
* Array offset
* Starting location
* Start of a log range
* Unix timestamp

Better:

```json
{
  "start_time": {
    "type": "string",
    "format": "date-time",
    "description": "Inclusive beginning of the search interval in ISO 8601 format, including timezone, such as 2026-07-14T09:00:00+05:30."
  }
}
```

Good parameter contracts reduce malformed calls and ambiguity.

---

# Prefer explicit types

Use constrained types where possible.

Instead of:

```json
{
  "priority": {
    "type": "string"
  }
}
```

Use:

```json
{
  "priority": {
    "type": "string",
    "enum": ["low", "medium", "high", "critical"]
  }
}
```

Instead of:

```json
{
  "count": {
    "type": "number"
  }
}
```

Use:

```json
{
  "count": {
    "type": "integer",
    "minimum": 1,
    "maximum": 100
  }
}
```

Instead of asking the model to create a generic configuration object:

```json
{
  "configuration": {
    "type": "object"
  }
}
```

Define its fields:

```json
{
  "configuration": {
    "type": "object",
    "properties": {
      "timeout_seconds": {
        "type": "integer",
        "minimum": 1,
        "maximum": 30
      },
      "include_archived": {
        "type": "boolean"
      }
    },
    "required": ["timeout_seconds"],
    "additionalProperties": false
  }
}
```

---

# Distinguish optional values from missing information

Suppose you expose:

```json
{
  "name": "book_meeting",
  "parameters": {
    "type": "object",
    "properties": {
      "attendee_email": {
        "type": "string"
      },
      "start_time": {
        "type": "string"
      },
      "duration_minutes": {
        "type": "integer"
      }
    },
    "required": [
      "attendee_email",
      "start_time",
      "duration_minutes"
    ]
  }
}
```

The model should not invent missing values.

If the user says:

> Set up a meeting with John.

The model lacks:

* Which John
* Date and time
* Duration

Depending on your instructions, the model should either:

* Ask the user for the missing data
* Call a contact-search tool first
* Call a calendar-availability tool
* Avoid calling the booking tool until required fields are known

A useful system instruction is:

```text
Do not guess required tool arguments. If a required value cannot be
derived from conversation context or another tool, ask the user.
```

---

# Results should also have a stable schema

Developers often carefully define the input contract but return arbitrary output.

For example, sometimes the tool returns:

```json
{
  "status": "done"
}
```

Sometimes:

```json
{
  "message": "Success"
}
```

Sometimes:

```text
Operation completed
```

This makes the model's job harder.

Use a predictable envelope:

```json
{
  "ok": true,
  "data": {},
  "error": null,
  "metadata": {}
}
```

For success:

```json
{
  "ok": true,
  "data": {
    "order_id": "ORD-12345",
    "status": "shipped"
  },
  "error": null,
  "metadata": {
    "source": "order_service",
    "retrieved_at": "2026-07-14T15:20:00+05:30"
  }
}
```

For failure:

```json
{
  "ok": false,
  "data": null,
  "error": {
    "code": "ORDER_NOT_FOUND",
    "message": "No order was found for the supplied order ID.",
    "retryable": false
  },
  "metadata": {}
}
```

This is much easier for the LLM to reason about than raw stack traces.

---

# Do not expose internal errors directly

Bad result:

```json
{
  "error": "psycopg2.OperationalError at db-prod-7.internal:5432 password authentication failed for user admin"
}
```

This may leak:

* Hostnames
* Database technology
* Usernames
* Infrastructure details
* Secrets
* Stack traces

Better:

```json
{
  "ok": false,
  "error": {
    "code": "DEPENDENCY_UNAVAILABLE",
    "message": "The order service is temporarily unavailable.",
    "retryable": true
  }
}
```

Keep the detailed error in application logs, not in the model context.

---

# Tool calls can be read-only or side-effecting

This distinction should be part of your contract.

## Read-only tool

```text
get_order_status
search_documents
get_weather
calculate_tax
```

These can often execute immediately.

## Side-effecting tool

```text
cancel_order
send_email
delete_file
transfer_money
deploy_application
```

These need stronger controls.

For a destructive tool, consider a two-step contract:

```text
1. prepare_order_cancellation
2. confirm_order_cancellation
```

Example first result:

```json
{
  "ok": true,
  "data": {
    "confirmation_token": "cancel_abc123",
    "order_id": "ORD-12345",
    "refund_amount": 1499.00,
    "warning": "This action cannot be undone."
  }
}
```

The user confirms, then the second tool executes:

```json
{
  "confirmation_token": "cancel_abc123"
}
```

Do not rely solely on the LLM to decide whether a high-impact action is authorized.

---

# Idempotency should be part of write-tool design

LLM calls, network requests, and orchestrator loops can be retried.

Suppose this tool is called twice:

```json
{
  "name": "charge_customer",
  "arguments": {
    "customer_id": "CUS-1",
    "amount": 5000
  }
}
```

Without idempotency, the customer might be charged twice.

Use an idempotency key:

```json
{
  "name": "charge_customer",
  "arguments": {
    "customer_id": "CUS-1",
    "amount": 5000,
    "idempotency_key": "payment-order-12345"
  }
}
```

Your backend guarantees that repeated executions using the same key return the original outcome rather than repeating the side effect.

This is especially important for:

* Payments
* Email sending
* Ticket creation
* Deployments
* Order cancellation
* File mutation

---

# Parallel tool calls

A model may request multiple independent tools in one response.

For example:

```json
{
  "tool_calls": [
    {
      "id": "call_weather",
      "name": "get_weather",
      "arguments": {
        "city": "Bengaluru"
      }
    },
    {
      "id": "call_calendar",
      "name": "get_calendar_events",
      "arguments": {
        "date": "2026-07-15"
      }
    }
  ]
}
```

Your orchestrator can execute them concurrently:

```python
results = await asyncio.gather(
    execute_tool(weather_call),
    execute_tool(calendar_call),
)
```

It then sends two separate tool-result messages, each linked through its call ID.

Parallel calls are useful when neither call depends on the other's output.

---

# Dependent tool calls require separate rounds

Consider:

> Find Sai's most recent order and tell me whether it has shipped.

The model may need:

```text
Round 1:
find_customer(name="Sai")

Result:
customer_id="CUS-91"

Round 2:
list_customer_orders(customer_id="CUS-91", limit=1)

Result:
order_id="ORD-12345"

Round 3:
get_order_status(order_id="ORD-12345")

Result:
status="shipped"
```

Because later arguments depend on earlier results, they cannot all necessarily be created at once.

A composite tool could reduce this:

```text
get_latest_customer_order_status(customer_name="Sai")
```

But that creates a more specialized contract.

The choice is a trade-off:

```text
Small atomic tools:
+ Reusable
+ Flexible
- More model round trips
- Greater latency

Composite tools:
+ Fewer round trips
+ Lower latency
+ Easier for common workflows
- Less reusable
- More backend complexity
```

A good production system often exposes both:

```text
Atomic:
- find_customer
- list_orders
- get_order_status

Composite:
- get_latest_order_summary
```

---

# The tool contract is not necessarily your HTTP contract

Suppose your internal service exposes:

```http
GET /api/v3/customer/order/query?customerReference=...
```

You do not need to expose this exact interface to the model.

The LLM-facing tool can be cleaner:

```json
{
  "name": "find_customer_orders",
  "parameters": {
    "type": "object",
    "properties": {
      "customer_id": {
        "type": "string"
      }
    }
  }
}
```

Your adapter translates it:

```python
def find_customer_orders(customer_id: str) -> dict:
    response = http_client.get(
        "/api/v3/customer/order/query",
        params={"customerReference": customer_id},
    )

    return normalize_order_response(response.json())
```

This creates an anti-corruption layer:

```text
LLM schema
    ↓
Tool adapter
    ↓
Internal API, database, subprocess, or Rust binary
```

The model does not need to know how your internal systems are structured.

---

# A Python implementation pattern

Here is a simplified contract-first design.

```python
from __future__ import annotations

from typing import Any, Callable
from pydantic import BaseModel, Field, ValidationError


class GetOrderStatusInput(BaseModel):
    order_id: str = Field(
        min_length=5,
        max_length=50,
        pattern=r"^ORD-[0-9]+$",
    )
    include_history: bool = False


class ToolExecutionError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.retryable = retryable


def get_order_status(
    order_id: str,
    include_history: bool = False,
) -> dict[str, Any]:
    # Replace this with a database, HTTP call, Rust FFI, etc.
    if order_id != "ORD-12345":
        raise ToolExecutionError(
            code="ORDER_NOT_FOUND",
            message="No order was found for the supplied order ID.",
            retryable=False,
        )

    result: dict[str, Any] = {
        "order_id": order_id,
        "status": "shipped",
        "carrier": "DHL",
        "estimated_delivery": "2026-07-16",
    }

    if include_history:
        result["history"] = [
            {
                "status": "processing",
                "timestamp": "2026-07-13T10:00:00+05:30",
            },
            {
                "status": "shipped",
                "timestamp": "2026-07-14T09:00:00+05:30",
            },
        ]

    return result


class ToolDefinition:
    def __init__(
        self,
        *,
        input_model: type[BaseModel],
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        self.input_model = input_model
        self.handler = handler


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "get_order_status": ToolDefinition(
        input_model=GetOrderStatusInput,
        handler=get_order_status,
    )
}


def execute_tool(
    tool_name: str,
    raw_arguments: dict[str, Any],
) -> dict[str, Any]:
    definition = TOOL_REGISTRY.get(tool_name)

    if definition is None:
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "UNKNOWN_TOOL",
                "message": "The requested tool is unavailable.",
                "retryable": False,
            },
        }

    try:
        validated = definition.input_model.model_validate(raw_arguments)

        result = definition.handler(**validated.model_dump())

        return {
            "ok": True,
            "data": result,
            "error": None,
        }

    except ValidationError as exc:
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "INVALID_ARGUMENTS",
                "message": "The tool arguments were invalid.",
                "retryable": False,
                "details": exc.errors(
                    include_url=False,
                    include_input=False,
                ),
            },
        }

    except ToolExecutionError as exc:
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": exc.code,
                "message": exc.safe_message,
                "retryable": exc.retryable,
            },
        }

    except Exception:
        # Log the actual exception internally.
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "INTERNAL_TOOL_ERROR",
                "message": "The tool could not complete the request.",
                "retryable": True,
            },
        }
```

The model-facing schema can be generated from the Pydantic model:

```python
tool_schema = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": (
            "Get the current shipping and delivery status of an order. "
            "Use this when the user asks whether an order has shipped, "
            "is delayed, or has been delivered."
        ),
        "parameters": GetOrderStatusInput.model_json_schema(),
    },
}
```

This reduces the chance that your Python validation and model-facing schema drift apart.

---

# The contract should contain execution metadata outside the LLM schema

Some information should exist in your internal registry but not necessarily be exposed to the model.

```python
class ToolMetadata:
    name: str
    timeout_seconds: int
    read_only: bool
    requires_confirmation: bool
    required_permission: str
    max_result_bytes: int
    allow_parallel: bool
    retry_policy: str
```

Example:

```python
TOOL_METADATA = {
    "get_order_status": {
        "timeout_seconds": 5,
        "read_only": True,
        "requires_confirmation": False,
        "required_permission": "orders:read",
        "max_result_bytes": 20_000,
        "allow_parallel": True,
        "retry_policy": "safe",
    },
    "cancel_order": {
        "timeout_seconds": 10,
        "read_only": False,
        "requires_confirmation": True,
        "required_permission": "orders:cancel",
        "max_result_bytes": 10_000,
        "allow_parallel": False,
        "retry_policy": "idempotency-key-required",
    },
}
```

The model describes what it wants. Your runtime determines whether it is safe and permitted.

---

# Contract versioning

Tool schemas change over time.

For example, version 1:

```json
{
  "name": "get_weather",
  "parameters": {
    "city": "string"
  }
}
```

Version 2:

```json
{
  "name": "get_weather",
  "parameters": {
    "location": {
      "city": "string",
      "country_code": "string"
    }
  }
}
```

Changing the schema while old applications still expect the previous format can break execution.

Possible approaches:

### Version the tool name

```text
get_weather_v1
get_weather_v2
```

### Keep the public tool stable

```text
get_weather
```

Then have your adapter support both old and new input formats.

### Version the tool registry

```text
tool_catalog_version: 2026-07-01
```

For most systems, keeping the LLM-facing contract stable and adapting internally is preferable.

---

# What the LLM contract cannot guarantee

Even with a strict schema, the LLM can still:

* Select the wrong tool
* Supply the wrong but structurally valid value
* Call a tool unnecessarily
* Fail to call a required tool
* Misinterpret the tool result
* Retry a non-retryable operation
* Attempt tool-call loops

For example:

```json
{
  "order_id": "ORD-99999"
}
```

This may be syntactically valid while referring to the wrong order.

Therefore, schemas provide **structural reliability**, not complete semantic correctness.

Reliability comes from combining:

```text
Good schema
+ Good descriptions
+ Runtime validation
+ Authorization
+ Confirmation for side effects
+ Tool-result normalization
+ Limits and timeouts
+ Evaluation and monitoring
```

---

# Practical contract-design rules

For each tool, define:

```text
1. Name
   Clear verb + object:
   get_order_status
   search_documents
   create_support_ticket

2. Purpose
   What business capability does it expose?

3. Selection rules
   When should the model use it?
   When should it not?

4. Input schema
   Types, required fields, ranges, enums, formats.

5. Authorization
   What user or service permissions are required?

6. Side effects
   Is it read-only, mutating, destructive, or externally visible?

7. Confirmation policy
   Must the user confirm before execution?

8. Idempotency
   Can it safely be retried?

9. Timeout and retry policy
   How long may it run?
   Which failures are retryable?

10. Output schema
    What exact structured data is returned?

11. Error schema
    Stable error codes and safe messages.

12. Data limits
    Maximum output size, result count, and pagination strategy.

13. Observability
    Tool name, latency, success/failure, call ID, and error code.

14. Versioning
    How will schema changes be introduced?
```

The fundamental pattern is:

```text
The model-facing schema expresses intent.

The executor-facing implementation performs the operation.

The result schema communicates evidence back to the model.

The orchestrator validates, authorizes, and connects all three.
```


