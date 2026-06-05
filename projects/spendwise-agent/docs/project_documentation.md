# SpendWise Agent Project Documentation

## Project Overview

SpendWise Agent is an AI-powered personal expense tracker built as a local Streamlit application. The goal of the project is to help users upload real bank statements, clean and standardize transaction data, categorize expenses, view financial dashboards, and ask an AI chatbot questions about their spending.

The project started as a dashboard idea for sample expense data, but the main learning came from working with real bank statement PDFs. Real statements were much messier than sample data, so the project evolved into a full statement-to-insight workflow.

The final app supports:

- PDF and CSV bank statement upload.
- Transaction extraction and standardization.
- Merchant name cleaning.
- Category assignment using rules.
- Human review and correction of categories.
- Learned category rules stored locally.
- Dashboard metrics and spending charts.
- Overspending alerts.
- Subscription detection.
- Personalized savings insights.
- Floating AI chatbot.
- Multi-provider LLM support through LangChain.

## What Was Built

The application is organized as a portfolio-ready project under:

```text
projects/spendwise-agent/
```

Main files and folders:

```text
app.py                    Streamlit app entry point
src/ingestion.py          PDF/CSV parsing, cleaning, standardization, category rules
src/analytics.py          Metrics, alerts, subscriptions, insights
src/ai.py                 AI chatbot and helper prompts
src/llm.py                LangChain provider registry
src/styles.py             Custom UI styling
data/category_rules.csv   Local learned category memory
docs/                     Architecture and project documentation
tests/                    Unit tests
```

The app uses a clean transaction schema:

```text
date
merchant
category
amount
source
```

It also tracks `category_source` so the user can understand whether a category came from a learned rule, built-in rule, source category, or manual review.

## Datasets Used

This project used three types of data during development.

### 1. Sample Data

Sample transaction data was created inside the project so the UI could work before uploading real statements. This helped test the dashboard, charts, insights, and chatbot without depending on private financial data.

File:

```text
src/sample_data.py
```

Purpose:

- Validate UI layout.
- Test charts and metrics.
- Demonstrate the app without private data.
- Build initial dashboard behavior quickly.

### 2. Sample PDF

A sample PDF statement was generated for testing PDF upload behavior.

File:

```text
sample_expenses.pdf
```

Purpose:

- Test PDF extraction.
- Confirm that uploaded PDF data could flow into the same dashboard pipeline.
- Provide a safe demo file for GitHub.

### 3. Real Bank Statements

Real statement PDFs were used locally during development, including statements such as January-February and April-May transaction periods.

These real files are intentionally not committed to GitHub. They are ignored through `.gitignore` because they may contain private financial data.

Purpose:

- Test real-world PDF parsing.
- Find categorization gaps.
- Validate the need for cleaning and learned rules.
- Improve merchant matching and category logic.

## Vibe Coding Prompts Used

The project was developed through iterative prompt-driven coding. These are the main types of prompts used during the workflow.

### Initial App Prompt

The first prompt described a financial dashboard UI:

```text
Create a UI that acts as an expense tracker agent with dashboard, analytics, subscriptions, overspending alerts, insights, and a floating AI chatbot.
```

This helped define the first version of the product experience.

### Architecture Prompt

An architecture screenshot was provided, and the app was refined around:

```text
Top navigation, upload control, month selector, multiple tabs, and an always-visible floating AI chatbot.
```

This prompt shaped the overall layout and user flow.

### Implementation Prompt

A detailed implementation plan was provided:

```text
Build a local single-user Streamlit app named SmartSpend with polished dark dashboard UI, top controls for PDF/CSV upload and month selection, tabs, and an OpenAI-powered floating chat panel.
```

This became the first full implementation direction.

### Provider Switching Prompt

After OpenAI quota limits, the project moved to a model-agnostic setup:

```text
Use Groq, Gemini, OpenAI, and Ollama through LangChain so the model can be swapped in and out.
```

This led to the LLM provider registry in `src/llm.py`.

### Real Statement Improvement Prompt

The main product improvement came from this problem:

```text
The sample data works perfectly, but real bank statements are not categorizing properly. I want to cleanse and standardize the PDF first, convert into CSV format, then do calculations.
```

This shifted the app from a simple dashboard into a real ingestion and review workflow.

### Learning Rules Prompt

After seeing repeated uncategorized rows, the workflow evolved again:

```text
How do I handle changing real data when every new statement has different merchants and missing categories?
```

This led to the learned category rules system using `data/category_rules.csv`.

### Portfolio Prompt

The project was then reorganized for GitHub and portfolio presentation:

```text
Create a proper structure to upload into GitHub and make it portfolio-friendly instead of week1/week2 naming.
```

This led to the `projects/spendwise-agent/` structure and portfolio documentation.

## Iterations Tried

### Iteration 1: Dashboard With Sample Data

The first working version focused on UI and analytics using sample transactions.

What worked:

- Dashboard cards rendered correctly.
- Charts worked.
- Insights worked.
- Chatbot UI worked.

Limitation:

- Sample data was already clean, so it did not expose real ingestion problems.

### Iteration 2: PDF Upload

PDF upload was added using `pdfplumber`.

What worked:

- Text and table extraction worked for some statements.
- Basic PDF parsing could extract transaction-like rows.

Limitation:

- Real bank PDF layouts were inconsistent.
- Some rows were missed or parsed incorrectly.
- Merchant names were noisy.

### Iteration 3: Standardized CSV Output

The workflow was adjusted to first cleanse and standardize PDF data before using it for analytics.

What improved:

- Dashboard calculations became more reliable.
- Transactions were easier to inspect.
- The user could download a standardized CSV.

Limitation:

- Categorization still missed new merchants from older or different statements.

### Iteration 4: Built-In Merchant Rules

Merchant aliases and built-in keyword rules were added.

What improved:

- Common merchants like Publix, Target, Uber, Netflix, and Starbucks were categorized more accurately.
- Noise like card numbers, location names, and payment words was cleaned.

Limitation:

- Hardcoded rules could not cover every real merchant.

### Iteration 5: Human Review and Learned Rules

The Import tab was expanded with an editable review table and a Save category rules button.

What improved:

- User corrections became reusable.
- Future statements could use saved merchant-category patterns.
- Categorization became adaptive without a database or model training.

This became the most important design improvement in the project.

### Iteration 6: Multi-Provider LLM Support

The AI layer was refactored to use LangChain and multiple providers.

What improved:

- The app no longer depended only on OpenAI.
- Groq, Gemini, OpenAI, and Ollama could be swapped through `.env`.
- Ollama gave a local open-weight option.

### Iteration 7: Portfolio Structure and Documentation

The repo was organized as a portfolio project.

What improved:

- Clear GitHub structure.
- Architecture diagrams.
- Slide briefing.
- Project documentation.
- Better README.

## Issues Faced and Resolutions

### Issue 1: LLM Token Limits, Quota Limits, and Model Switching

During development, the app initially depended on a single LLM provider. This worked for early testing, but it created problems when the API quota was exceeded or when a model became unavailable.

Problem:

- The OpenAI API hit quota/token limits during testing.
- Switching to another model or provider required touching multiple files.
- Provider-specific code was mixed into the app logic.
- Every model change felt like a code rewrite instead of a configuration change.

Resolution:

The AI layer was refactored into a LangChain-based orchestration layer.

The project now has a provider registry in:

```text
src/llm.py
```

The app can switch between providers by changing `.env` values instead of rewriting the application.

Supported providers:

- Groq
- Gemini
- OpenAI
- Ollama

Example:

```text
AI_PROVIDER=groq
```

or:

```text
AI_PROVIDER=ollama
```

This made the project more flexible and resilient. If one provider runs out of quota or a model is unavailable, another provider can be plugged in with minimal code change.

Learning:

For agentic applications, the LLM should be treated as a swappable service, not hardcoded into the business logic. LangChain helped separate orchestration from the Streamlit UI and analytics code.

### Issue 2: Real PDF Statements Created Too Many Uncategorized Transactions

The app worked well with sample data because the sample transactions were clean and predictable. However, when real bank statements were uploaded, many transactions were marked as `Uncategorized`.

Problem:

- Real merchant names were noisy and inconsistent.
- PDF extraction included location names, card codes, numbers, and extra text.
- Different statements introduced new merchants that the model or built-in rules had not seen before.
- Relying only on the LLM for categorization was not reliable or cost-efficient.
- Re-uploading older or different statements exposed new gaps in categorization.

Resolution:

A rule-based golden dataset was created using:

```text
data/category_rules.csv
```

This file acts as local learned memory. When the user reviews transactions and corrects categories, the app saves merchant-category mappings into the CSV file.

Example:

```csv
pattern,merchant,category
publix,Publix,Groceries
target,Target,Shopping
netflix,Netflix,Subscriptions
om indian market,Om Indian Market,Groceries
```

The categorization flow became:

```text
1. Check learned rules from category_rules.csv
2. Use source category if available
3. Use built-in merchant keyword rules
4. Fall back to Uncategorized
```

This means the app improves over time. If a new real statement contains a merchant that was corrected before, the saved rule can categorize it automatically.

Learning:

For real financial data, a pure LLM approach is not enough. A hybrid approach works better:

```text
Human review + learned rules + built-in rules + optional LLM fallback
```

This makes the categorization more explainable, cheaper to run, and more stable across changing real-world statements.

## Key Learnings

### 1. Sample Data Can Hide Real-World Problems

The app looked strong with sample data, but real bank statements exposed parsing and categorization issues. This showed that financial apps need to be tested with realistic messy data early.

### 2. Data Cleaning Is More Important Than Charts

The dashboard is only useful if the underlying transaction data is clean. The project became stronger after focusing on ingestion, standardization, and review before analytics.

### 3. Human-in-the-Loop Review Is Valuable

AI and rules can help, but the user still needs a way to correct mistakes. The review table made the system more trustworthy.

### 4. Learned Rules Are Practical Memory

Instead of relying only on an LLM to categorize every transaction, the app stores user corrections locally. This is simple, explainable, and improves future uploads.

### 5. LLM Provider Flexibility Matters

Using only one provider can cause problems with quotas, billing, or model availability. Moving to LangChain made the AI layer easier to switch between Groq, Gemini, OpenAI, and Ollama.

### 6. Local Privacy Needs Clear Boundaries

Financial data is sensitive. The project avoids committing real PDFs, `.env` files, extracted text, and standardized CSVs. Only safe sample files and reusable category rules are kept in the repo.

### 7. Architecture Evolves With Real User Workflow

The original idea was a dashboard with tabs. The final architecture became a pipeline:

```text
Upload → Extract → Clean → Categorize → Review → Learn → Analyze → Chat
```

This better matches how a real user would trust the output.

## Observations From the Workflow

- The hardest part was not creating the UI; it was handling real statement variability.
- PDF extraction needs fallback strategies because every bank formats statements differently.
- Categorization should not be treated as a one-time prediction problem.
- A small local rules file can be very effective for personal finance use cases.
- User trust improves when the app shows extracted rows before calculating insights.
- AI chat is most useful after the transaction data has already been standardized.
- Local open-weight models are useful when API quota or privacy is a concern.

## Current Limitations

- Benchmarks are hardcoded and not yet personalized by income or goals.
- PDF parsing is heuristic and may fail on some statement layouts.
- There is no persistent database in v1.
- Category rules are local to one machine.
- Subscription detection is heuristic.
- The app does not connect directly to bank APIs.
- The app is local single-user only.

## Future Improvements

- Add SQLite storage for historical uploads.
- Add a category rules management screen.
- Add personalized budget settings.
- Add income-aware financial health scoring.
- Add better duplicate detection.
- Add stronger PDF extraction using layout-aware document AI.
- Add a dedicated Analytics tab.
- Add a dedicated Subscriptions tab.
- Add a dedicated Alerts tab.
- Add local-only privacy mode using Ollama.
- Add exportable PDF/HTML monthly reports.

## Final Summary

SpendWise Agent is a practical AI finance assistant that turns raw bank statements into useful financial insights. The strongest part of the project is the workflow design: it does not assume uploaded data is clean. Instead, it extracts, standardizes, reviews, learns, and then analyzes.

For a demo or portfolio, the key message is:

```text
SpendWise Agent is not just an expense dashboard. It is a statement-to-insight agent that learns from real user corrections and makes messy financial data understandable.
```
