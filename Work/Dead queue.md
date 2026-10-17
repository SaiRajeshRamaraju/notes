# Task: Azure Service Bus DLQ Notifications

The objective is to set up an automated notification system when messages fail to process and are moved to the Dead Letter Queue (DLQ) in Azure Service Bus. Since the DLQ operates as a sub-queue , standard Azure consumer patterns can be attached to it to dispatch alerts via Microsoft Teams or Email.

If we want to use the alerts that comes with serviec bus, the current alerts as of now doesn't support them or we need to add logs for dlq to work with the azure alerts.

Below are the three primary architectures to achieve this, detailing their flows, pros, and cons.

## Option 1: Azure Functions (Code-Driven) or Azure Logic apps

An **Azure Function** is configured with a `ServiceBusTrigger` targeting the `$DeadLetterQueue` path. It extracts message metadata (like `DeadLetterReason`), formats a custom payload, and sends it via an HTTP POST request to a Teams Webhook or via Azure Communication Services for email.

### Pros and Cons

- **Pros:** Highly customizable. You can filter out transient errors, parse specific metadata, and precisely format the alert payload.
- **Cons:** Requires writing, deploying, and maintaining custom code. You must also implement strict error handling to avoid infinite DLQ loops if the notification logic fails.

A standard Logic App is created with a Service Bus trigger targeting the `$DeadLetterQueue`. Using built-in connectors for Microsoft Teams and Office 365 Outlook, the workflow dynamically maps the message properties into Adaptive Cards or email bodies without writing deployment code.

### Pros and Cons

- **Pros:** Rapid integration with zero code to maintain. Provides native, built-in connectors with rich visual formatting for Teams and Outlook.
- **Cons:** Can become expensive if DLQ volume is extremely high. Version controlling the ARM/Bicep templates for Logic Apps can be verbose and complex.

## Option 2: Azure Monitor Alerts (Threshold-Based)

This isn't useful for our case atleast. Azure Monitor is configured with an Alert Rule targeting the `DeadletteredMessages` metric on the Service Bus namespace. When the count exceeds a static threshold (e.g., &gt; 5 messages in 5 minutes), an Action Group is triggered to dispatch an email or hit a Teams webhook.

### Pros and Cons

- **Pros:** Excellent for general infrastructure health monitoring. Prevents alert fatigue by notifying on volume rather than per message.
- **Cons:** Lacks granularity. Notifications do not include the message payload, Message ID, or the specific exception reason; manual inspection of the queue is still required.

## Option 3 : Use Azure Alerts (Service Bus Alerts)

**P.S: Not entirely sure if this works or not**Azure services bus have pre backed alerts. By default we can't use the alert for dead quee since there aren't proper logs for it. Now update the code to have dead queue logs , to add log when a new message comes into dlq.

### Pros and Cons

- **Pros** : Using prexisting alerts that comes with azure service bus without adding extra layyer for alerts.
- **Cons** : Not entire sure if this works or not or how to do this. We already added multiple logs,metrics and traces using telemetry but none of them appeared in the logs options. So need to dig deepr