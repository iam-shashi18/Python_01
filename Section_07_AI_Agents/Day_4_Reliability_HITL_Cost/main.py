# hitl_delete_user.py

# Install first:
# pip install langgraph

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 1. FAKE DATABASE
# ============================================================
# This is only for learning.
# No real database is involved.

users_db = {
    "U101": {
        "name": "John",
        "email": "john@example.com",
        "role": "Developer",
    },
    "U102": {
        "name": "Sarah",
        "email": "sarah@example.com",
        "role": "Manager",
    },
    "U103": {
        "name": "David",
        "email": "david@example.com",
        "role": "Admin",
    },
}


def show_users():
    print("\n========== USERS DATABASE ==========")

    for user_id, user in users_db.items():
        print(
            f"{user_id} | "
            f"{user['name']} | "
            f"{user['email']} | "
            f"{user['role']}"
        )

    print("====================================")


# ============================================================
# 2. LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict):
    user_request: str
    user_id: str
    user_details: str
    human_decision: str
    result: str


# ============================================================
# 3. NODE: UNDERSTAND USER REQUEST
# ============================================================

def understand_request(state: AgentState):

    print("\n🤖 AI: Understanding request")

    request = state["user_request"]

    print("User:", request)

    # For this beginner example we simply extract
    # the first word starting with U.

    user_id = None

    for word in request.split():

        if word.upper().startswith("U"):
            user_id = word.upper()
            break

    if user_id is None:

        return {
            "user_id": "",
            "result": "Could not identify user."
        }

    print("AI identified user:", user_id)

    return {
        "user_id": user_id
    }


# ============================================================
# 4. TOOL: GET USER DETAILS
# ============================================================
# Reading data is considered safe.
# Therefore, no human approval is required here.

def get_user_details(state: AgentState):

    print("\n🔍 TOOL: get_user_details()")

    user_id = state["user_id"]

    if user_id not in users_db:

        return {
            "user_details": "",
            "result": f"User {user_id} does not exist."
        }

    user = users_db[user_id]

    details = (
        f"User ID : {user_id}\n"
        f"Name    : {user['name']}\n"
        f"Email   : {user['email']}\n"
        f"Role    : {user['role']}"
    )

    print(details)

    return {
        "user_details": details
    }


# ============================================================
# 5. TOOL: DELETE USER
# ============================================================
#
# THIS IS THE IMPORTANT PART.
#
# We put interrupt() BEFORE the destructive operation.
#
# ============================================================

def delete_user(state: AgentState):

    user_id = state["user_id"]

    print("\n⚠️ TOOL: delete_user()")

    print("\nThe AI wants to delete:")
    print("--------------------------------")
    print(state["user_details"])
    print("--------------------------------")

    # ========================================================
    # HUMAN-IN-THE-LOOP
    # ========================================================
    #
    # LangGraph pauses HERE.
    #
    # The user has NOT been deleted yet.
    #

    human_decision = interrupt(
        {
            "type": "DELETE_USER_APPROVAL",
            "message": "Approve deleting this user?",
            "user_id": user_id,
            "user_details": state["user_details"],
        }
    )

    # ========================================================
    # LangGraph resumes here after approval
    # ========================================================

    print("\n👤 Human decision:", human_decision)

    if human_decision == "approve":

        # ====================================================
        # ACTUAL DELETE OPERATION
        # ====================================================

        if user_id in users_db:

            del users_db[user_id]

            print(
                f"\n🗑️ User {user_id} DELETED."
            )

            return {
                "human_decision": "approve",
                "result": f"User {user_id} successfully deleted."
            }

        return {
            "human_decision": "approve",
            "result": f"User {user_id} does not exist."
        }

    else:

        print(
            f"\n❌ Deletion of {user_id} rejected."
        )

        return {
            "human_decision": "reject",
            "result": f"Deletion of {user_id} rejected by human."
        }


# ============================================================
# 6. CREATE LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)


# Add nodes

graph.add_node(
    "understand_request",
    understand_request
)

graph.add_node(
    "get_user_details",
    get_user_details
)

graph.add_node(
    "delete_user",
    delete_user
)


# ============================================================
# 7. CONNECT NODES
# ============================================================

graph.add_edge(
    START,
    "understand_request"
)

graph.add_edge(
    "understand_request",
    "get_user_details"
)

graph.add_edge(
    "get_user_details",
    "delete_user"
)

graph.add_edge(
    "delete_user",
    END
)


# ============================================================
# 8. CHECKPOINTING
# ============================================================
#
# The checkpointer remembers the state when the graph
# is interrupted.
#

checkpointer = InMemorySaver()

app = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 9. USER REQUEST
# ============================================================

user_request = "Delete U102"

print("\n")
print("=" * 60)
print("USER REQUEST")
print("=" * 60)

print(user_request)


# ============================================================
# 10. INITIAL STATE
# ============================================================

initial_state = {
    "user_request": user_request,
    "user_id": "",
    "user_details": "",
    "human_decision": "",
    "result": "",
}


# ============================================================
# 11. THREAD ID
# ============================================================
#
# Think of this as the ID of this particular workflow.
#
# If there are 1,000 deletion requests, each one gets
# a different thread_id.
#

config = {
    "configurable": {
        "thread_id": "delete-request-001"
    }
}


# ============================================================
# 12. START WORKFLOW
# ============================================================

print("\n")
print("=" * 60)
print("STARTING LANGGRAPH")
print("=" * 60)

result = app.invoke(
    initial_state,
    config=config
)


# ============================================================
# 13. GRAPH IS NOW PAUSED
# ============================================================
#
# Because delete_user() called interrupt().
#
# IMPORTANT:
#
# Nothing has been deleted yet.
#
# The Python execution that called app.invoke() has returned
# because LangGraph is waiting for external input.
#

print("\n")
print("=" * 60)
print("⏸️ LANGGRAPH PAUSED")
print("=" * 60)

print(
    """
LangGraph reached interrupt().

The user has NOT been deleted.

The application now needs to ask a human
for approval.
"""
)


# ============================================================
# 14. GET PAUSED STATE
# ============================================================

current_state = app.get_state(config)

print("\n📌 PAUSED STATE")
print("--------------------------------")

for key, value in current_state.values.items():

    print(f"{key}: {value}")


# ============================================================
# 15. APPLICATION ASKS HUMAN
# ============================================================
#
# IMPORTANT:
#
# This input() is NOT LangGraph.
#
# This is our APPLICATION asking the human.
#
# In a real application this could instead be:
#
#   Web UI
#   REST API
#   Teams
#   Slack
#   Admin dashboard
#
# ============================================================

print("\n")
print("=" * 60)
print("👤 HUMAN APPROVAL REQUIRED")
print("=" * 60)

print("\nThe following user will be deleted:\n")

print(
    current_state.values["user_details"]
)

print()

human_decision = input(
    "Approve deletion? (approve/reject): "
).lower().strip()


# ============================================================
# 16. RESUME LANGGRAPH
# ============================================================
#
# This sends the human's decision back into LangGraph.
#
# Example:
#
# Command(resume="approve")
#
# LangGraph continues from the interrupt().
#
# ============================================================

print("\n▶️ Sending human decision to LangGraph...")

result = app.invoke(
    Command(
        resume=human_decision
    ),
    config=config
)


# ============================================================
# 17. FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(result["result"])


# ============================================================
# 18. SHOW DATABASE
# ============================================================

show_users()


# ============================================================
# FINAL FLOW
# ============================================================

print("\n")
print("=" * 60)
print("COMPLETE FLOW")
print("=" * 60)

print(
"""
User
 │
 │ "Delete user U102"
 ↓
┌────────────────────────┐
│ understand_request     │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ get_user_details()     │
│                        │
│ Safe READ operation    │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ delete_user()          │
│                        │
│ Dangerous operation    │
└───────────┬────────────┘
            ↓
       interrupt()
            │
            ↓
        ⏸️ PAUSED
            │
            ↓
     Application asks
        human 👤
            │
       ┌────┴────┐
       ↓         ↓
    approve    reject
       │         │
       ↓         ↓
   DELETE      STOP
       │
       ↓
      END


IMPORTANT:

interrupt()
    =
"Pause the workflow."

input()
    =
"Application asks the human."

Command(resume=...)
    =
"Send human's answer back to LangGraph."

delete operation
    =
"Only happens after approval."
"""
)