from __future__ import annotations
from typing import Dict, Any, List, Literal
from dataclasses import dataclass
import json, os
import sys
from pathlib import Path

# Add the src directory to the path so we can import from scorer
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from scorer import score_invoice  # NEW: real scorer using D2 model

@dataclass
class MatchResult:
    status: Literal["match", "partial", "mismatch"]
    confidence: float
    facts: Dict[str, Any]
    explanation: str

def call_matcher(invoice_id: str, po_number: str, data_dir: str, model_path: str) -> MatchResult:
    res = score_invoice(data_dir=data_dir, model_path=model_path, invoice_id=invoice_id, po_number=po_number)
    if not res.get("found"):
        return MatchResult(status="partial", confidence=0.5, facts={}, explanation="No features available for this pair.")
    
    # Get raw model output (probability of mismatch)
    mismatch_prob = float(res["confidence"])
    facts = res["facts"]
    
    # Determine status based on multiple factors, not just model threshold
    has_material_issues = (
        facts.get("po_missing", False) or
        not facts.get("vendor_match", True) or
        facts.get("amount_delta", 0) > 0.01 or
        not facts.get("has_grn", True)
    )
    
    # Enhanced status determination with better thresholds to reduce false positives
    if has_material_issues:
        status = "mismatch"
        confidence = mismatch_prob
    elif mismatch_prob >= 0.8:  # Higher threshold for mismatch (was 0.7)
        status = "mismatch"
        confidence = mismatch_prob
    elif mismatch_prob >= 0.6:  # Higher threshold for partial (was 0.3)
        status = "partial"
        confidence = mismatch_prob
    else:  # Lower mismatch probability - treat as match
        status = "match"
        confidence = 1.0 - mismatch_prob  # Convert to match confidence
    
    explanation = build_explanation(facts, status, confidence)
    return MatchResult(status=status, confidence=confidence, facts=facts, explanation=explanation)
def build_explanation(facts: Dict[str, Any], status: str, confidence: float) -> str:
    """Build comprehensive explanation based on facts and model output."""
    issues = []
    if facts.get("po_missing"):
        issues.append("PO reference was not found")
    if not facts.get("vendor_match", True):
        issues.append("Vendor on invoice does not match vendor on PO")
    if facts.get("amount_delta", 0) > 0.01:
        issues.append(f"Amount discrepancy of {facts['amount_delta']:.2f}")
    if not facts.get("has_grn", True):
        issues.append("No GRN found for this PO")
    
    # Add timing concerns if significant
    days_delta = abs(facts.get("days_delta", 0))
    if days_delta > 30:
        issues.append(f"Invoice timing concern: {days_delta} days from GRN")
    
    if status == "mismatch":
        if issues:
            return f"Mismatch detected: {'; '.join(issues)}. Confidence: {confidence:.2f}"
        else:
            return f"Model detected potential mismatch (confidence: {confidence:.2f}). Manual review recommended."
    elif status == "partial":
        return f"Uncertain match requiring review (confidence: {confidence:.2f}). {'; '.join(issues) if issues else 'Model suggests possible issues.'}"
    else:
        if issues:
            return f"Match confirmed despite minor issues: {'; '.join(issues)}. Confidence: {confidence:.2f}"
        else:
            return f"Clean match with no material differences. Confidence: {confidence:.2f}"

# ---- LLM draft email tool (pseudo) ----

def draft_dispute_email(vendor_name: str, invoice_id: str, po_number: str, facts: Dict[str, Any], match_status: str) -> str:
    """Generate context-aware dispute email based on actual issues found."""
    
    # Identify specific issues
    issues = []
    if facts.get("po_missing"):
        issues.append("• PO reference could not be located in our system")
    if not facts.get("vendor_match", True):
        issues.append("• Vendor information does not match our PO records")
    if facts.get("amount_delta", 0) > 0.01:
        amount_delta = facts.get("amount_delta", 0)
        issues.append(f"• Amount discrepancy of ${amount_delta:.2f} detected")
    if not facts.get("has_grn", True):
        issues.append("• No goods receipt (GRN) found for the referenced PO")
    
    # Add timing concerns if significant
    days_delta = abs(facts.get("days_delta", 0))
    if days_delta > 30:
        issues.append(f"• Timing discrepancy: Invoice received {days_delta} days after goods receipt")
    
    # Choose email tone based on match status
    if match_status == "mismatch":
        subject_prefix = "URGENT: Invoice Discrepancy"
        urgency_text = "We have identified significant discrepancies that require immediate attention:"
        action_text = "Please provide corrected documentation or explanation within 5 business days. Payment is currently on hold."
    elif match_status == "partial":
        subject_prefix = "Review Required"
        urgency_text = "We are reviewing your invoice and need clarification on the following items:"
        action_text = "Please review and provide supporting documentation or confirmation within 7 business days."
    else:
        subject_prefix = "Clarification Requested"
        urgency_text = "We are processing your invoice and would appreciate clarification on minor items:"
        action_text = "Please provide any additional documentation at your convenience. This will not delay payment processing."
    
    # Build the email
    issues_text = "\n".join(issues) if issues else "• General compliance review as part of our standard process"
    
    email_body = f"""Subject: {subject_prefix} - Invoice {invoice_id} / PO {po_number}

Dear {vendor_name},

{urgency_text}

{issues_text}

Invoice Details:
- Invoice ID: {invoice_id}
- PO Number: {po_number}
- Amount Delta: ${facts.get('amount_delta', 0):.2f}
- Vendor Match: {'Yes' if facts.get('vendor_match', True) else 'No'}
- GRN Available: {'Yes' if facts.get('has_grn', True) else 'No'}
- PO Status: {'Found' if not facts.get('po_missing', False) else 'Missing'}

{action_text}

If you have any questions, please contact our Accounts Payable team at ap@acme-manufacturing.com or call (555) 123-4567.

Best regards,
Accounts Payable Department
Acme Manufacturing
"""
    return email_body

# ---- Guardrails ----
ALLOWED_TOOLS = {"matcher": call_matcher, "email_drafter": draft_dispute_email}

def guardrail_tool_call(tool_name: str, *args, **kwargs):
    """Enhanced guardrails with validation."""
    assert tool_name in ALLOWED_TOOLS, f"Tool '{tool_name}' not permitted. Allowed: {list(ALLOWED_TOOLS.keys())}"
    
    # Additional validation for specific tools
    if tool_name == "matcher":
        assert len(args) >= 4, "Matcher requires: invoice_id, po_number, data_dir, model_path"
        assert all(isinstance(arg, str) for arg in args[:4]), "All matcher args must be strings"
    
    elif tool_name == "email_drafter":
        assert len(args) >= 5, "Email drafter requires: vendor_name, invoice_id, po_number, facts, status"
        assert isinstance(args[3], dict), "Facts must be a dictionary"
        assert args[4] in ["match", "partial", "mismatch"], f"Invalid status: {args[4]}"
    
    try:
        result = ALLOWED_TOOLS[tool_name](*args, **kwargs)
        return result
    except Exception as e:
        raise RuntimeError(f"Tool '{tool_name}' execution failed: {str(e)}") from e

def planner_node(context: Dict[str, Any]) -> Dict[str, Any]:
    # expand invoices in batch to tasks
    tasks = []
    for item in context.get("invoices", []):
        tasks.append({"invoice_id": item["invoice_id"],
                      "po_number": item["po_number"],
                      "vendor_name": item["vendor_name"]})
    return {"tasks": tasks}

def reconcile_node(context: Dict[str, Any]) -> Dict[str, Any]:
    """Perform matching and determine which invoices need email follow-up."""
    results = []
    for t in context["tasks"]:
        mr = guardrail_tool_call("matcher", t["invoice_id"], t["po_number"], context["data_dir"], context["model_path"])
        t["match_result"] = mr.__dict__
        
        # Improved email trigger logic - only for actual issues
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

def email_node(context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate dispute emails for invoices that need follow-up."""
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
            t["email_draft"] = None  # No email needed for clean matches
    return {"tasks": context["tasks"]}

def approval_gate(context: Dict[str, Any]) -> Dict[str, Any]:
    """Human approval checkpoint with summary statistics."""
    # Generate summary statistics
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
    
    # STOP here for human approval in UI/workflow engine
    context["status"] = "APPROVAL_AWAITING" if context["summary"]["approval_required"] else "COMPLETED"
    return context

def agent_run(data_dir: str, model_path: str, invoices: List[Dict[str, Any]]):
    context = {"data_dir": data_dir, "model_path": model_path, "invoices": invoices, "min_conf": 0.75}
    context.update(planner_node(context))
    context.update(reconcile_node(context))
    context.update(email_node(context))
    context = approval_gate(context)
    return context

if __name__ == "__main__":
    print("Starting agent_graph.py...")
    
    demo_invoices = [
        {"invoice_id": "INV0012", "po_number": "PO0012", "vendor_name": "Acme Supplies Ltd"},
        {"invoice_id": "INV0199", "po_number": "PO0199", "vendor_name": "Global Parts Co"},
        {"invoice_id": "INV0001", "po_number": "PO0001", "vendor_name": "Test Vendor Co"},
    ]
    print(f"Demo invoices: {demo_invoices}")
    try:
        out = agent_run(data_dir="./data", model_path="./reports/matcher_model.pkl", invoices=demo_invoices)
        print("Agent run completed successfully!")
        print(json.dumps(out, indent=2))
    except Exception as e:
        print(f"Error during agent run: {e}")
        import traceback
        traceback.print_exc()
