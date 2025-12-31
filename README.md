## Agentic Home Energy Management System

This repository demonstrates an **agent-based AI system for home energy optimization**, built using **LangGraph** to coordinate multiple AI agents working together to solve a real-world problem: reducing unnecessary power consumption while maintaining safety and transparency.

The system models energy management as a collaborative workflow where each LangGraph node represents a specialized agent with a clearly defined responsibility.

### How It Works

- **Perception Agent (`perceive`)**  
  Interprets raw electrical signals (power, power factor, time) to detect which appliances are active and converts them into a structured environment state.

- **Decision Agent (`brain`)**  
  Uses a Q-learning policy to recommend energy-saving actions based on historical outcomes and current context.

- **Safety Agent (`safety`)**  
  Enforces rule-based constraints, overriding unsafe or undesirable decisions (e.g., time-based appliance restrictions).

- **Execution & Learning Agent (`act`)**  
  Executes the final action, computes rewards, and updates the reinforcement learning model.

- **Notification Agent (`email`)**  
  Simulates user notifications when optimization actions occur.

- **Explainability Agent (`llm_analyst`)**  
  Uses a local LLM to explain why an action was taken and provides a concise energy-saving tip.

### Architecture

The agents are orchestrated using a **LangGraph `StateGraph`** with a shared typed state:

```
Perceive → Brain → Safety → Act → Notify → Explain → END
```