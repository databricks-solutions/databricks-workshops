# Vibe-to-Value (V2V) Workshop

This directory contains the assets for the Vibe-to-Value (V2V) workshop.

## Contents

| Asset | Description |
|-------|-------------|
| `Vibing_to_Value_Seattle.py` | Guide for the hands-on build day in the Databricks Seattle office on 11/17/26. It contains the session slides and the build prompts for three tracks: Retail, Field Service, and Helpdesk. |
| `customer_360_lead_scoring_sales.py` | Sales notebook (Customer 360, Lead Scoring, Churn & Upsell Prediction). It joins CRM, billing, product telemetry and support data into one governed customer view in Unity Catalog. AI Functions turn raw text into signals. MLflow tracks the lead, churn and upsell models. Genie and an AI/BI dashboard show the results. |
| `fpa_cash_forecasting_cfo.py` | Office of the CFO notebook (FP&A Variance, Cash Forecasting & Revenue Assurance). It explains plan variance, forecasts cash position, and finds revenue leakage between contracts and billing. AI Functions run in SQL. MLflow validates the numbers. Genie and an AI/BI dashboard show the results. |
| `document_intelligence_financial_services.py` | Financial Services notebook (Loan, Investment & Claims Document Intelligence). It builds a multi-task Lakeflow Job of AI Function tasks. The tasks parse document packets, classify and split pages, and extract loan, deal and claim fields. The notebook then reconciles the fields against systems of record, flags mismatches, and enables Genie. |
| `document_intelligence_manufacturing.py` | Manufacturing notebook (Order, Invoice & Engineering Document Intelligence). It converts purchase orders, invoices, product data sheets and quality documents into validated data. The data includes line items, prices, specs and measurements. The notebook flags exceptions and enables Genie. |
| `patient_demand_claims_healthcare.py` | Healthcare notebook (Patient Demand, Workforce & Claims Intelligence). It forecasts patient demand, predicts no-shows, finds staffing gaps, and predicts claim denials. All steps use Databricks AI Functions in SQL. Genie answers questions on the results. |

All industry notebooks are Databricks source-format notebooks. Each notebook starts with a setup section. Widgets in that section set the catalog and schema for the demo tables.

## How to get help

Databricks support does not cover this content. For questions or bugs, open a GitHub issue. The team will help on a best effort basis.
