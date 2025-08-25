# SmartPay AP - Accounts Payable Automation Platform

This repository contains my solution for the SmartPay AP technical assessment, implementing a next-generation Agentic AI platform for accounts payable automation at Acme Manufacturing.

## Overview

Acme Manufacturing processes approximately 1 million supplier invoices per month across 25 countries. This solution addresses their need for an intelligent system that can extract data from invoices, match against purchase orders, and automate dispute resolution workflows while maintaining compliance with GDPR requirements.

The implementation demonstrates a minimal viable solution that combines machine learning, agentic workflows, and enterprise-grade architecture principles to meet the stated business requirements.

## Solution Architecture

The platform consists of three main components:

**Data Processing Pipeline**: Handles invoice and purchase order data ingestion, normalization, and feature engineering. The system processes CSV data representing invoices, purchase orders, and goods receipt notes with realistic business scenarios including missing POs, price variances, and vendor mismatches.

**Machine Learning Model**: A logistic regression classifier that identifies invoice-PO mismatches with precision-focused optimization to minimize false positives. The model incorporates 12 key features including vendor matching, amount variances, date validations, and business rule compliance checks.

**Agentic Workflow**: An intelligent agent built using a state machine pattern that plans reconciliation steps, calls the ML model for matching decisions, generates contextual dispute emails, and implements human approval gates for sensitive actions.

## Getting Started

### Prerequisites

The solution requires Python 3.8 or higher with the following key dependencies:
- pandas for data manipulation
- scikit-learn for machine learning
- numpy for numerical computations

### Installation

Clone the repository and set up your environment:

```bash
git clone <repository-url>
cd SmartPay_AP_D2
pip install -r requirements.txt
```

### Running the Solution

To train the matching model and generate evaluation metrics:

```bash
python -m src.cli --data-dir ./data --out-dir ./reports
```

To execute the complete agentic workflow:

```bash
python src/agent/agent_graph.py
```

The agent will process demo invoices and demonstrate the complete workflow including matching decisions, email generation, and approval requirements.

## Key Features

**Intelligent Invoice Matching**: The system uses a combination of exact matching and fuzzy logic to link invoices with purchase orders. It handles real-world scenarios like vendor name variations, currency mismatches, and missing purchase order references.

**Risk-Based Decision Making**: Rather than relying solely on model predictions, the system implements multi-factor decision logic that considers business rules, materiality thresholds, and confidence levels to determine appropriate actions.

**Automated Dispute Resolution**: When mismatches are detected, the system generates contextually appropriate emails with varying urgency levels based on the severity of discrepancies. All communications include detailed supporting information and appropriate escalation timelines.

**Governance and Compliance**: The platform implements a comprehensive governance framework with multiple layers of protection. This includes tool allowlisting to restrict agent actions, input validation with type checking and parameter validation, exception handling with detailed error context, and business logic constraints that prevent over-reliance on single model outputs. The system maintains detailed decision audit trails, implements risk-based escalation workflows, and requires mandatory human approval for all email communications and sensitive operations.

## Technical Implementation

### Data Engineering

The feature engineering pipeline creates meaningful business indicators from raw invoice and PO data. Key features include vendor similarity scoring, amount variance calculations with percentage and absolute thresholds, temporal relationship validation, and compliance flag generation.

The system handles missing data gracefully and applies realistic business tolerances to avoid excessive false positives while maintaining appropriate risk controls.

### Machine Learning

The classification model uses L1-regularized logistic regression with carefully tuned class weights to balance precision and recall. The model focuses on minimizing false positives to prevent overwhelming users with unnecessary reviews while ensuring genuine mismatches are flagged appropriately.

Feature importance analysis reveals that late payment flags, vendor matching, and missing PO indicators are the strongest predictors of mismatches, aligning with expected business patterns.

### Agentic Architecture

The workflow implementation follows a four-stage process: planning, reconciliation, email generation, and approval. Each stage includes appropriate error handling and validation to ensure reliable operation.

The agent incorporates several guardrails including tool allowlisting, input validation, exception handling, and business logic constraints. Human approval gates ensure that no automated actions occur without explicit authorization.

## Governance and Risk Management

### Guardrails Implementation

The system implements a defense-in-depth approach to ensure safe and compliant operation:

**Tool Access Control**: The agent can only execute pre-approved functions through a strict allowlist mechanism. Any attempt to use unauthorized tools results in immediate termination with detailed error logging.

**Input Validation**: All tool calls undergo comprehensive validation including parameter count verification, data type checking, and business rule compliance. Invalid inputs are rejected with descriptive error messages before any processing occurs.

**Exception Handling**: Robust error handling ensures graceful degradation when issues arise. All exceptions are caught, logged with full context, and converted to actionable error messages while preserving system stability.

**Multi-Factor Decision Logic**: Rather than relying solely on model predictions, the system combines ML outputs with business rules, confidence thresholds, and materiality assessments to make final decisions.

### Compliance Features

**Audit Trail**: Every decision includes comprehensive documentation of the reasoning process, confidence scores, and supporting evidence. This creates a complete audit trail for regulatory compliance and quality assurance.

**Human Approval Gates**: Critical actions such as email generation and payment holds require explicit human authorization. The system provides summary statistics and risk assessments to support decision-making.

**Status Tracking**: Clear workflow state management ensures all processes can be monitored, audited, and controlled. Each invoice maintains detailed status information throughout the reconciliation process.

**Risk-Based Escalation**: The system automatically categorizes issues by severity and routes them through appropriate escalation channels with different urgency levels and approval requirements.

## Performance Metrics

The current model achieves the following performance on the test dataset:
- Precision: 1.000 (no false positives)
- Recall: 0.400 (conservative approach)
- F1-Score: 0.571

These metrics reflect a deliberately conservative approach prioritizing precision over recall to minimize operational disruption while ensuring compliance requirements are met.

## Project Structure

```
SmartPay_AP_D2/
├── data/                    # Sample dataset files
├── reports/                 # Generated metrics and model artifacts
├── src/
│   ├── agent/              # Agentic workflow implementation
│   ├── data.py             # Data loading and normalization
│   ├── features.py         # Feature engineering pipeline
│   ├── model.py            # ML training and evaluation
│   └── cli.py              # Command-line interface
├── requirements.txt
└── README.md
```

## Design Decisions

**Conservative Threshold Selection**: The system uses higher confidence thresholds than typical classification problems to minimize false positives. This approach prioritizes operational efficiency and user trust over maximum automation.

**Governance-First Design**: All components include built-in safeguards and approval mechanisms. The system is designed to err on the side of caution, ensuring that automated actions only occur when confidence levels are extremely high and human oversight is maintained.

**Modular Architecture**: Each component can be tested, validated, and deployed independently. This design supports iterative improvement and makes the system easier to maintain and extend while ensuring changes can be audited and controlled.

**Realistic Business Logic**: Rather than purely academic ML optimization, the solution incorporates practical business considerations like payment terms, vendor relationship management, and regulatory compliance requirements.

## Future Enhancements

**Advanced NLP Integration**: Incorporate natural language processing for invoice description matching and automated classification of mismatch types.

**Multi-Currency Support**: Expand currency handling to include real-time exchange rate validation and multi-currency reconciliation logic.

**Integration Capabilities**: Develop APIs for integration with SAP, Oracle, and other enterprise systems commonly used in accounts payable operations.

**Enhanced Monitoring**: Implement comprehensive model performance monitoring, data drift detection, and automated retraining workflows for production deployment.

## Assumptions and Limitations

The current implementation assumes a simplified invoice-PO relationship model suitable for the provided synthetic dataset. Production deployment would require more sophisticated entity resolution and data quality validation.

The solution focuses on the core business logic and ML components rather than user interface development, though the email generation templates demonstrate the expected user experience.

Performance optimization has not been prioritized for this prototype, though the modular design supports scaling improvements as needed for production workloads.

## Contact Information

This solution was developed as part of a technical assessment and demonstrates practical application of machine learning, workflow automation, and enterprise software design principles in the accounts payable domain.
