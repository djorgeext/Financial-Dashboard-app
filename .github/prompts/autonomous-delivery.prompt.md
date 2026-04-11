---
name: Autonomous Delivery
description: "Run implementation and review autonomously through delivery-orchestrator."
argument-hint: "Describe the goal, constraints, and acceptance criteria."
agent: "delivery-orchestrator"
model: "Claude Haiku 4.5 (copilot)"
tools: [agent, todo, read, search, edit, execute]
---

Execute this task end-to-end with minimal user intervention.

Requirements:
- Delegate code changes to implementation-agent.
- Delegate code review to reviewer-agent.
- Keep all operations inside /home/david/Documents/trade.
- Ask the user only if blocked by ambiguity, missing credentials, or boundary policy.
- Return final status, changed files, validation results, unresolved risks, and explicit handoff notes if blocked.
