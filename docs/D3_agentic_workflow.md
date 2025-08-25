# Agentic Workflow Sketch for SmartPay AP

This document outlines the agentic workflow for invoice-PO reconciliation using the `agent_graph.py` and supporting modules. The workflow leverages modular agent nodes, tool guardrails, and LLM-powered explanations and email drafting.

## 1. Workflow Overview

The agentic workflow is implemented as a sequence of nodes/functions, each responsible for a specific step in the reconciliation process:

1. **Planner Node**: Expands the input invoice list into actionable tasks.
2. **Reconcile Node**: Calls the matching model for each invoice/PO pair and determines if follow-up is needed.
3. **Email Node**: Generates a draft dispute email for mismatches or partial matches using an LLM.
4. **Approval Gate**: Summarizes results and stops for human approval before proceeding.

## 2. Key Components & Code Snippets

### a. Planner Node
```python
def planner_node(context: Dict[str, Any]) -> Dict[str, Any]:
    tasks = []
    for item in context.get("invoices", []):
        tasks.append({
            "invoice_id": item["invoice_id"],
            "po_number": item["po_number"],
            "vendor_name": item["vendor_name"]
        })
    return {"tasks": tasks}
```

### b. Reconcile Node (Model Tool Call)
```python
def reconcile_node(context: Dict[str, Any]) -> Dict[str, Any]:
    results = []
    for t in context["tasks"]:
        mr = guardrail_tool_call("matcher", t["invoice_id"], t["po_number"], context["data_dir"], context["model_path"])
        t["match_result"] = mr.__dict__
        # Email trigger logic
        needs_email = False
        if mr.status == "mismatch":
            needs_email = True
            t["email_reason"] = "Material discrepancies detected"
        elif mr.status == "partial":
            needs_email = True
            t["email_reason"] = "Uncertain match requires clarification"
        elif mr.confidence < context.get("min_conf", 0.75):
            needs_email = True
            t["email_reason"] = "Low confidence match requires review"
        else:
            t["email_reason"] = "Clean match - no action required"
        t["needs_email"] = needs_email
        results.append(t)
    return {"tasks": results}
```

### c. Email Node (LLM Draft)
```python
def email_node(context: Dict[str, Any]) -> Dict[str, Any]:
    for t in context["tasks"]:
        if t.get("needs_email"):
            t["email_draft"] = guardrail_tool_call(
                "email_drafter",
                t["vendor_name"],
                t["invoice_id"],
                t["po_number"],
                t["match_result"]["facts"],
                t["match_result"]["status"]
            )
        else:
            t["email_draft"] = None
    return {"tasks": context["tasks"]}
```

### d. Approval Gate (Human-in-the-loop)
```python
def approval_gate(context: Dict[str, Any]) -> Dict[str, Any]:
    total_invoices = len(context["tasks"])
    matches = sum(1 for t in context["tasks"] if t["match_result"]["status"] == "match")
    partial_matches = sum(1 for t in context["tasks"] if t["match_result"]["status"] == "partial")
    mismatches = sum(1 for t in context["tasks"] if t["match_result"]["status"] == "mismatch")
    emails_needed = sum(1 for t in context["tasks"] if t.get("needs_email", False))
    context["summary"] = {
        "total_invoices": total_invoices,
        "clean_matches": matches,
        "partial_matches": partial_matches,
        "mismatches": mismatches,
        "emails_to_send": emails_needed,
        "approval_required": emails_needed > 0 or mismatches > 0
    }
    context["status"] = "APPROVAL_AWAITING" if context["summary"]["approval_required"] else "COMPLETED"
    return context
```

## 3. Tool Guardrails

All agent tool calls are routed through a guardrail function:
```python
def guardrail_tool_call(tool_name: str, *args, **kwargs):
    assert tool_name in ALLOWED_TOOLS, f"Tool '{tool_name}' not permitted."
    # Additional validation for each tool...
    return ALLOWED_TOOLS[tool_name](*args, **kwargs)
```
This ensures only approved tools (model matcher, email drafter) are invoked, with argument validation.

## 4. End-to-End Agent Run

```python
def agent_run(data_dir: str, model_path: str, invoices: List[Dict[str, Any]]):
    context = {"data_dir": data_dir, "model_path": model_path, "invoices": invoices, "min_conf": 0.75}
    context.update(planner_node(context))
    context.update(reconcile_node(context))
    context.update(email_node(context))
    context = approval_gate(context)
    return context
```

## 5. Human Approval Step

The workflow stops at the approval gate for human review before any emails are sent or payments are triggered, ensuring responsible AI and compliance.

---

**References:**
- See `src/agent/agent_graph.py` for full implementation.
- Guardrails and tool validation are essential for safe agentic automation.
