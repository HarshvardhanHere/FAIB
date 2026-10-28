# Bharat Horizon Bank — Fictional Home Loan Policy

**Document type:** Educational prototype policy  
**Scope:** Fictional home-loan applications for a college project. This is not a real bank policy or financial advice.

## 1. Basic eligibility
- The applicant's monthly income must be at least ₹25,000.
- Applications must include the documents applicable to the applicant's employment category.
- Missing required information or documents must be sent for manual review rather than treated as proof of ineligibility.

## 2. Credit score
- Credit score of 750 or above: meets the credit-score criterion.
- Credit score from 650 through 749: manual review.
- Credit score below 650: rejection recommendation under this prototype policy.

## 3. Fixed-obligation-to-income ratio (FOIR)
For this prototype, FOIR is calculated as:

`FOIR = (existing monthly EMI + estimated proposed monthly EMI) / monthly income × 100`

- FOIR of 40% or below: meets the FOIR criterion.
- FOIR above 40% and up to 50%: manual review.
- FOIR above 50%: rejection recommendation under this prototype policy.

The dataset's FOIR calculation uses an illustrative 8.5% annual interest rate. This is a demonstration assumption, not an offered or current loan rate.

## 4. Loan amount relative to income
- The loan amount must not exceed 2 times the applicant's annual income.
- A ratio above 2.0 requires a rejection recommendation under this prototype policy.
- Annual income is monthly income multiplied by 12.

## 5. Past defaults
- A default in the last 12 months leads to a rejection recommendation under this prototype policy.
- A default older than 12 months requires manual review.
- If the timing of a past default is missing or unclear, refer the application for manual review.

## 6. Employment category and documents
- Salaried applicants must provide the applicable identity, income/salary, employment, and property documents.
- Self-employed applicants must provide the applicable identity, income/business, financial, and property documents.
- This prototype policy does not define an exhaustive document checklist. If required documents cannot be confirmed, send the case for manual review.

## 7. Manual review
A human reviewer must review an application when any of these apply:
- Credit score is 650–749.
- FOIR is above 40% and up to 50%.
- A past default is older than 12 months, or the timing is unclear.
- Required documents are missing or cannot be verified.
- The model probability is between 0.40 and 0.60 inclusive.
- Policy clauses conflict, the case is ambiguous, or the policy does not address the scenario.
- The system's model recommendation conflicts with a policy-based rejection indicator.

## 8. Policy gaps
The policy does not specify a waiting period or eligibility treatment for a **recently settled loan default** when the settlement date and default timing do not fit the rules above. Do not invent a rule. Explain that the policy does not cover the case and escalate it to a human reviewer.

## 9. Human authority
The model's prediction and any LLM-generated explanation are advisory only. They must not automatically approve or reject an application. A human reviewer retains final decision authority and can record an override with a reason.

## 10. Safety and use
This is a fictional educational prototype using synthetic data. It is not a real lending policy and must not be used for real credit decisions or as financial, legal, or other professional advice.
