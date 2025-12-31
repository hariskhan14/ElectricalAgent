import operator
import random
import numpy as np
from typing import TypedDict, List, Dict
from collections import defaultdict
from langgraph.graph import StateGraph, END

# LLM Imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

ACTIONS = ["DO_NOTHING", "TURN_OFF_IRON", "TURN_OFF_BULBS", "TURN_OFF_FAN", "TURN_OFF_DC_FAN"]
POWER_MAP = {"IRON": 1000, "REF": 150, "FAN": 75, "3 BULBS": 180, "DC FAN": 60}
LABELS = ['IRON', 'REF', 'FAN', '3 BULBS', 'DC FAN']

# Initialize Local LLM
llm = ChatOllama(model="llama3.2", temperature=0.7)

# ==========================================
# Q-LEARNING AGENT
# ==========================================

class QAgent:
    def __init__(self, n_actions):
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=float))
        self.alpha = 0.3
        self.gamma = 0.9
        self.epsilon = 0.1

    def choose(self, s_key):
        print(f"[QAgent] Choosing action for state_key: {s_key}")
        if random.random() < self.epsilon:
            action = random.randrange(len(self.Q[s_key]))
            print(f"[QAgent] Random action chosen: {ACTIONS[action]}")
            return action
        action = int(np.argmax(self.Q[s_key]))
        print(f"[QAgent] Best action chosen: {ACTIONS[action]}")
        return action

    def learn(self, s_key, a_idx, reward, s2_key):
        print(f"[QAgent] Learning update")
        print(f"  State: {s_key}")
        print(f"  Action: {ACTIONS[a_idx]}")
        print(f"  Reward: {reward}")
        best_next = float(np.max(self.Q[s2_key]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[s_key][a_idx]
        self.Q[s_key][a_idx] += self.alpha * td_error
        print(f"  Updated Q[{s_key}][{ACTIONS[a_idx]}] = {self.Q[s_key][a_idx]}")

brain_agent = QAgent(n_actions=len(ACTIONS))

# ==========================================
# GRAPH STATE
# ==========================================

class AgentState(TypedDict):
    Vrms: float
    P: float
    PF: float
    hour: int
    current_appliances: Dict[str, bool]
    state_key: tuple
    recommended_action: str
    executed_action: str
    reward: float
    llm_analysis: str

# ==========================================
# GRAPH NODES
# ==========================================

def perceive_node(state: AgentState):
    print("\n[PERCEIVE NODE] ----------------------------")
    print("Input State:", state)

    p = state['P']
    appliances = {
        'IRON': p > 700,
        'REF': p > 80,
        '3 BULBS': p > 150,
        'FAN': p > 120 and p <= 250,
        'DC FAN': p > 200
    }

    hour_bin = state['hour'] // 4
    pf_bin = 0 if state['PF'] < 0.6 else (1 if state['PF'] < 0.8 else (2 if state['PF'] < 0.9 else 3))
    p_bin = 0 if p < 100 else (1 if p < 300 else (2 if p < 800 else 3))
    bits = tuple(int(appliances[l]) for l in LABELS)

    output = {
        "current_appliances": appliances,
        "state_key": (bits, hour_bin, pf_bin, p_bin)
    }

    print("Detected Appliances:", appliances)
    print("Generated state_key:", output["state_key"])
    return output


def brain_node(state: AgentState):
    print("\n[BRAIN NODE] -------------------------------")
    print("State Key Received:", state['state_key'])

    s_key = state['state_key']
    a_idx = brain_agent.choose(s_key)

    output = {"recommended_action": ACTIONS[a_idx]}
    print("Recommended Action:", output["recommended_action"])
    return output


def safety_node(state: AgentState):
    print("\n[SAFETY NODE] ------------------------------")
    print("Recommended Action:", state['recommended_action'])
    print("Current Appliances:", state['current_appliances'])
    print("Hour:", state['hour'])

    rec = state['recommended_action']
    apps = state['current_appliances']
    hour = state['hour']

    final = rec
    if "TURN_OFF_REF" in rec:
        final = "DO_NOTHING"
    if hour >= 22 and apps.get("IRON", False):
        final = "TURN_OFF_IRON"
    if 6 <= hour <= 18 and apps.get("3 BULBS", False):
        final = "TURN_OFF_BULBS"

    print("Executed Action (After Safety):", final)
    return {"executed_action": final}


def act_node(state: AgentState):
    print("\n[ACT NODE] --------------------------------")
    print("Executed Action:", state['executed_action'])
    print("Previous Appliances:", state['current_appliances'])

    prev = state['current_appliances']
    action = state['executed_action']

    new_apps = prev.copy()
    if action == "TURN_OFF_IRON":
        new_apps["IRON"] = False
    elif action == "TURN_OFF_BULBS":
        new_apps["3 BULBS"] = False

    reward = 0.0
    if state['hour'] >= 22 and prev.get("IRON") and not new_apps.get("IRON"):
        reward += 10.0

    print("Reward Calculated:", reward)

    brain_agent.learn(
        state['state_key'],
        ACTIONS.index(state['recommended_action']),
        reward,
        state['state_key']
    )

    return {"reward": reward}



def email_notifier_node(state: AgentState):
    """Prints email status if an optimization action was taken."""
    print("\n[Email Notify NODE] -------------------------")
    if state['executed_action'] != "DO_NOTHING":
        print(f"Email Sent: Notification sent to user regarding action: {state['executed_action']}")
    return {} 

def llm_analyst_node(state: AgentState):
    print("\n[LLM ANALYST NODE] -------------------------")
    print("Sending context to LLM...")

    prompt_text = """
    You are an expert Home Energy Manager AI.

    CURRENT SITUATION:
    - Time: {hour}:00
    - Active Power: {P} Watts
    - Detected Appliances ON: {appliances}

    DECISION LOG:
    - The RL Brain recommended: {recommended}
    - The Safety System executed: {executed}

    TASK:
    1. Analyze why this action was taken.
    2. Explain safety override if any.
    3. Give one energy-saving tip.

    Keep it concise.
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "hour": state['hour'],
        "P": state['P'],
        "appliances": [k for k, v in state['current_appliances'].items() if v],
        "recommended": state['recommended_action'],
        "executed": state['executed_action']
    })

    print("LLM Response Generated")
    return {"llm_analysis": response}

# ==========================================
# BUILD GRAPH
# ==========================================

print("\n[GRAPH] Building LangGraph workflow...")

workflow = StateGraph(AgentState)

workflow.add_node("perceive", perceive_node)
workflow.add_node("brain", brain_node)
workflow.add_node("safety", safety_node)
workflow.add_node("act", act_node)
workflow.add_node("email", email_notifier_node)
workflow.add_node("llm_analyst", llm_analyst_node)

workflow.set_entry_point("perceive")
workflow.add_edge("perceive", "brain")
workflow.add_edge("brain", "safety")
workflow.add_edge("safety", "act")
workflow.add_edge("act","email")
workflow.add_edge("email", "llm_analyst")
workflow.add_edge("llm_analyst", END)

app = workflow.compile()

# ==========================================
# RUN DEMO
# ==========================================

print("\n🚀 Running Agentic Energy System (DEBUG MODE)\n")

inputs = {
    "Vrms": 230.0,
    "P": 1100.0,
    "PF": 0.95,
    "hour": 23,
    "current_appliances": {},
    "state_key": (),
    "recommended_action": "",
    "executed_action": "",
    "reward": 0.0,
    "llm_analysis": ""
}

result = app.invoke(inputs)

print("\n================ FINAL OUTPUT ================")
print(f"Time: {result['hour']}:00")
print(f"Action Taken: {result['executed_action']}")
print(f"Reward: {result['reward']}")
print("---------------------------------------------")
print("LLM ANALYST REPORT:")
print(result['llm_analysis'])
print("=============================================")
