# AI Real Estate Sales & Rental Management Assistant

An AI-powered real estate assistant designed to support the end-to-end sales and rental lifecycle through conversational AI, RAG-based property discovery, lead qualification, viewing scheduling, CRM integration, and intelligent document processing.

The system is designed as a **production-oriented MVP** for real-estate companies that already have property information, documents, communication channels, and business systems, but need an intelligent AI layer that connects them into one conversational workflow.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Goals](#project-goals)
- [MVP Scope](#mvp-scope)
- [Out of Scope](#out-of-scope)
- [Core Workflow](#core-workflow)
- [System Architecture](#system-architecture)
- [Property Knowledge and RAG Architecture](#property-knowledge-and-rag-architecture)
- [Document Processing Workflow](#document-processing-workflow)
- [Customer Query Workflow](#customer-query-workflow)
- [Lead Qualification Workflow](#lead-qualification-workflow)
- [Viewing Scheduling Workflow](#viewing-scheduling-workflow)
- [CRM and Human Handoff](#crm-and-human-handoff)
- [Data Sources](#data-sources)
- [Data Model](#data-model)
- [Project Structure](#project-structure)
- [Technology Areas](#technology-areas)
- [AI Design Principles](#ai-design-principles)
- [Evaluation](#evaluation)
- [Security and Reliability](#security-and-reliability)
- [Observability](#observability)
- [MVP Constraints](#mvp-constraints)
- [8-Week Roadmap](#8-week-roadmap)
- [Running the Project](#running-the-project)
- [API](#api)
- [Testing](#testing)
- [Production Readiness Checklist](#production-readiness-checklist)
- [Future Enhancements](#future-enhancements)
- [Project Philosophy](#project-philosophy)

---

# Overview

The **AI Real Estate Sales & Rental Management Assistant** is an AI-powered system that acts as the first-line assistant for real-estate customers.

The assistant communicates with customers through:

- Website
- WhatsApp

and supports:

- Arabic
- English

The system understands customer requirements, retrieves suitable properties from the company's property knowledge base, recommends relevant units, qualifies leads, schedules property viewings, updates CRM systems, and performs human handoff when required.

The system also processes real-estate documents and images using:

- Document parsing
- OCR
- Computer Vision
- Information extraction
- Data normalization
- Embeddings
- RAG

The main goal is to build an **AI-powered real-estate workflow**, not just a chatbot.

---

# Problem Statement

Real-estate companies receive large volumes of customer inquiries across multiple communication channels.

Customers may ask about:

- Property availability
- Property prices
- Locations
- Areas
- Bedrooms
- Bathrooms
- Property types
- Payment plans
- Installments
- Projects
- Units
- Viewing appointments
- Buying or rental options

At the same time, property information may be distributed across multiple formats:

```text
PDF
Scanned PDF
DOCX
XLSX
CSV
Images
Brochures
Reservation Contracts
Unit Sketches
Floor Plans
```

Traditional workflows require sales representatives to manually:

- Search property information.
- Answer repetitive questions.
- Match customers with properties.
- Qualify leads.
- Schedule viewings.
- Update CRM systems.
- Follow up with customers.

This can lead to:

- Slow response times.
- Poor property matching.
- Lost leads.
- Repetitive manual work.
- Inconsistent customer experiences.
- Operational inefficiency.

The proposed solution introduces an AI layer that automates these operations while keeping critical business actions controlled, deterministic, and traceable.

---

# Project Goals

The system aims to:

- Improve customer response time.
- Improve property matching.
- Increase lead conversion.
- Automate repetitive customer interactions.
- Qualify and prioritize leads.
- Automate viewing scheduling.
- Integrate with existing CRM systems.
- Process unstructured real-estate documents.
- Support Arabic and English.
- Reduce hallucinated property information.
- Provide source-traceable recommendations.
- Improve sales-team productivity.
- Provide observability and evaluation.

## Business Targets

The target business outcomes are:

| KPI | Target |
|---|---:|
| Inquiry → Booking Conversion | +25% within first 6 months |
| Customer Acquisition Cost | -20% during first year |
| Customer Satisfaction | +15% within first 9 months |
| Residential Market Share | +5% over two years |

These are **business targets**, not guaranteed results.

---

# MVP Scope

## 1. Conversational AI Agent

The assistant should:

- Support Arabic and English.
- Understand customer intent.
- Extract customer requirements.
- Maintain conversational context.
- Answer property questions.
- Answer project questions.
- Answer price questions.
- Answer payment-plan questions.
- Ask clarification questions when needed.
- Provide grounded responses.
- Recommend relevant properties.
- Trigger business workflows when appropriate.

---

## 2. RAG-Based Property Discovery

Property discovery is primarily powered by a **RAG-based property knowledge system**.

The system should combine:

- Structured metadata filtering.
- Semantic search.
- Keyword / lexical retrieval where useful.
- Hybrid retrieval.
- Reranking.
- Grounded LLM generation.

The LLM should **not directly invent or independently select properties**.

Instead:

```text
Customer
   ↓
Requirement Extraction
   ↓
Structured Requirements
   ↓
Property Retrieval
   ↓
Metadata Filtering
   ↓
Semantic / Hybrid Search
   ↓
Reranking
   ↓
Relevant Properties
   ↓
LLM
   ↓
Final Recommendation
```

Example customer request:

> "عايز شقة في التجمع، 3 غرف، بحد أقصى 5 مليون، وتقسيط."

The LLM extracts:

```json
{
  "purpose": "sale",
  "location": "New Cairo",
  "property_type": "apartment",
  "bedrooms": 3,
  "max_price": 5000000,
  "payment_type": "installment"
}
```

The retrieval layer then searches the actual property knowledge base.

---

## 3. Lead Qualification

The system should:

- Identify buying or rental intent.
- Extract customer budget.
- Extract preferences.
- Identify purchase/rental timeline.
- Measure engagement.
- Estimate readiness.
- Calculate a lead score.
- Identify high-priority leads.
- Trigger human handoff.

For the MVP, lead scoring should be **deterministic and explainable**.

Example:

```text
Intent             25%
Budget             25%
Requirements       15%
Timeline           20%
Engagement         15%
-----------------------
Total              100%
```

Example:

```json
{
  "lead_score": 87,
  "lead_status": "hot",
  "readiness": "high"
}
```

---

## 4. Viewing Scheduling

The system should:

- Check representative availability.
- Retrieve available viewing slots.
- Present available slots.
- Confirm customer selection.
- Create the viewing.
- Link the viewing to the customer and property.
- Update the CRM when required.

---

## 5. CRM Integration

The system should integrate with the company's existing CRM where an API or supported integration is available.

Potential information includes:

- Customer profile.
- Lead profile.
- Customer requirements.
- Lead score.
- Recommended properties.
- Interaction history.
- Viewing information.
- Follow-up information.
- Human handoff information.

The MVP should **integrate with existing systems instead of replacing them**.

---

## 6. OCR and Computer Vision

The system supports intelligent processing of real-estate documents and visual content.

Potential inputs include:

- Scanned contracts.
- Property brochures.
- Unit sketches.
- Floor plans.
- Property images.
- Reservation documents.
- Scanned PDFs.

OCR and Computer Vision are **conditional processing capabilities**.

A document may require:

```text
Parser only
OCR only
CV only
OCR + CV
```

depending on its content.

---

# Out of Scope

The following capabilities are outside the MVP:

- Electronic payment integration.
- Advanced contract management.
- Custom mobile application.
- Replacing existing CRM/core systems.
- Advanced sentiment analysis.
- AR / immersive property experience.
- Predictive property pricing.
- Advanced market forecasting.
- Advanced risk prediction.
- Commercial real-estate expansion.
- Industrial real-estate expansion.

These capabilities may be introduced in future versions.

---

# Core Workflow

The project contains two major workflows:

1. **Property Knowledge Pipeline**
2. **Customer Business Workflow**

These workflows work together.

---

## Property Knowledge Pipeline

```text
Property Sources
       ↓
Document Type Detection
       ↓
Specialized Loading / Parsing
       ↓
OCR / Computer Vision when required
       ↓
Information Extraction
       ↓
Normalization
       ↓
Structured Metadata + Text
       ↓
Chunking
       ↓
Embeddings
       ↓
Vector Database
       ↓
Property Knowledge Base
```

---

## Customer Business Workflow

```text
Customer
   ↓
Website / WhatsApp
   ↓
API / Channel Layer
   ↓
Conversational AI Agent
   ↓
Intent Detection
   ↓
Requirement Extraction
   ↓
Property RAG Search
   ↓
Filtering + Hybrid Retrieval
   ↓
Reranking
   ↓
Property Recommendation
   ↓
Lead Qualification
   ↓
 ┌───────────────────────┐
 │                       │
 ↓                       ↓
Not Qualified          Qualified
 │                       │
 ↓                       ↓
Continue Chat            CRM
                         │
                  ┌──────┴──────┐
                  ↓             ↓
             Scheduling    Human Handoff
                  │             │
                  └──────┬──────┘
                         ↓
                  Sales Operation
```

---

# System Architecture

```text
                         ┌─────────────────────┐
                         │      Customer       │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ↓                     ↓
                    Website                WhatsApp
                         │                     │
                         └──────────┬──────────┘
                                    ↓
                          ┌───────────────────┐
                          │  API / Channels   │
                          └─────────┬─────────┘
                                    ↓
                          ┌───────────────────┐
                          │ Conversational AI │
                          │      Agent        │
                          └─────────┬─────────┘
                                    ↓
                         Intent + Requirements
                                    │
                ┌───────────────────┼───────────────────┐
                ↓                   ↓                   ↓
          Property RAG       Lead Qualification    Scheduling
                │                   │                   │
                ↓                   ↓                   ↓
          Retrieval +           Scoring +          Availability +
          Reranking              Handoff              Booking
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    ↓
                                  CRM
                                    │
                                    ↓
                           Sales Representative
```

---

# Property Knowledge and RAG Architecture

The property inventory is treated as a **knowledge source** that may come from multiple document and data formats.

The architecture does not assume that the property inventory is available through a dedicated developer API.

```text
                    PROPERTY SOURCES
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
       PDF             DOCX/Word          XLSX/CSV
        │                  │                  │
        ↓                  ↓                  ↓
   PDF Parser        Document Parser   Structured Parser
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ↓
                 CONTENT TYPE DETECTION
                           │
              ┌────────────┼────────────┐
              │            │            │
              ↓            ↓            ↓
        Text-Based      Scanned       Images /
        Documents       Documents     Floor Plans
              │            │            │
              │            ↓            │
              │           OCR           │
              │            │            │
              │            └─────┐      ↓
              │                  │      CV
              │                  │      │
              └──────────────────┴──────┘
                           │
                           ↓
                 INFORMATION EXTRACTION
                           │
                           ↓
                      NORMALIZATION
                           │
                ┌──────────┴──────────┐
                ↓                     ↓
        Structured Metadata           Text
                │                     │
                │                  Chunking
                │                     ↓
                │                 Embeddings
                │                     ↓
                │                  Vector DB
                │                     │
                └──────────┬──────────┘
                           ↓
                    HYBRID RETRIEVAL
                           ↓
                  METADATA FILTERING
                           ↓
                       RERANKER
                           ↓
                  RELEVANT PROPERTIES
                           ↓
                          LLM
                           ↓
                    RECOMMENDATION
```

---

# Document Processing Workflow

## PDF

For a normal text-based PDF:

```text
PDF
 ↓
PDF Parser
 ↓
Extracted Text
 ↓
Information Extraction
 ↓
Normalization
```

---

## Scanned PDF

For a scanned PDF:

```text
Scanned PDF
 ↓
OCR
 ↓
Extracted Text
 ↓
Information Extraction
 ↓
Normalization
```

---

## DOCX

```text
DOCX
 ↓
Document Parser
 ↓
Text / Tables
 ↓
Information Extraction
 ↓
Normalization
```

---

## XLSX / CSV

Structured data does not need traditional RAG processing first.

```text
XLSX / CSV
 ↓
Spreadsheet Parser
 ↓
Rows / Columns
 ↓
Field Mapping
 ↓
Normalization
 ↓
Property Metadata
```

The structured information can then be associated with the text/vector representation of the same property.

---

## Property Images

```text
Property Image
       │
       ├────────→ OCR
       │             ↓
       │       Text Information
       │
       └────────→ Computer Vision
                     ↓
               Visual Information
                     │
                     ↓
                Extraction
                     │
                     ↓
                Normalization
```

---

## Floor Plans / Unit Sketches

```text
Floor Plan
      │
      ├────────→ OCR
      │             ↓
      │       Labels / Numbers
      │
      └────────→ Computer Vision
                    ↓
              Layout / Structure
                    │
                    ↓
              Information Extraction
                    │
                    ↓
                 Normalization
```

---

# OCR vs Computer Vision

OCR and Computer Vision are not interchangeable.

## OCR

OCR is responsible for extracting text from visual content.

Examples:

- Unit numbers.
- Prices.
- Room labels.
- Contract text.
- Project names.
- Area values.
- Scanned documents.

## Computer Vision

Computer Vision is responsible for understanding visual information.

Examples:

- Floor-plan layouts.
- Room structures.
- Visual property characteristics.
- Spatial relationships.
- Unit sketches.

## Combined Processing

Some documents require both:

```text
Document
   ↓
OCR
   +
Computer Vision
   ↓
Combined Information
   ↓
Normalized Property Knowledge
```

---

# Retrieval Architecture

The retrieval system should not depend only on vector similarity.

Hard constraints should be represented using structured metadata whenever possible.

Examples:

```text
price <= 5,000,000
bedrooms = 3
location = New Cairo
purpose = sale
availability = available
```

Recommended retrieval flow:

```text
Customer Requirements
        ↓
Metadata Filters
        ↓
Candidate Properties
        ↓
Semantic / Hybrid Retrieval
        ↓
Reranking
        ↓
Top-K Properties
        ↓
LLM
```

This improves accuracy and reduces the chance of recommending properties that violate explicit customer requirements.

---

# Customer Query Workflow

Example:

> "عايز شقة 3 غرف في التجمع، معايا حوالي 4 مليون وممكن أقسط، وعايز أشتري الشهر الجاي."

## Step 1 — Intent Detection

```json
{
  "intent": "buy"
}
```

## Step 2 — Requirement Extraction

```json
{
  "intent": "buy",
  "location": "New Cairo",
  "property_type": "apartment",
  "bedrooms": 3,
  "budget": 4000000,
  "payment_preference": "installment",
  "purchase_timeline": "within_1_month"
}
```

## Step 3 — Retrieval

The system searches using:

- Location.
- Property type.
- Bedrooms.
- Budget.
- Payment preference.
- Semantic relevance.

## Step 4 — Reranking

Example:

```text
A102 → 0.94
B205 → 0.91
C301 → 0.87
```

## Step 5 — Grounded Recommendation

The LLM receives the retrieved property information and generates the response.

The model must not invent:

- Property IDs.
- Prices.
- Availability.
- Payment plans.
- Locations.

---

# Lead Qualification Workflow

```text
Customer Conversation
        ↓
Requirement Extraction
        ↓
Lead Qualification
        │
        ├── Intent
        ├── Budget
        ├── Requirements
        ├── Timeline
        └── Engagement
        ↓
Deterministic Scoring
        ↓
Lead Status
        │
   ┌────┼────────────┐
   ↓    ↓            ↓
 Cold  Warm         Hot
                    │
                    ↓
              Human Handoff
```

Example:

```json
{
  "customer_id": "C1001",
  "intent": "buy",
  "budget": 4500000,
  "location": "New Cairo",
  "property_type": "apartment",
  "bedrooms": 3,
  "timeline": "within_1_month",
  "lead_score": 87,
  "lead_status": "hot"
}
```

## Lead Scoring Principle

The LLM extracts information.

The scoring engine calculates the score.

```text
LLM
 ↓
Structured Lead Information
 ↓
Deterministic Scoring Engine
 ↓
Lead Score
```

This provides:

- Explainability.
- Consistency.
- Testability.
- Auditability.

---

# Viewing Scheduling Workflow

```text
Customer
   ↓
Viewing Request
   ↓
Identify Property
   ↓
Check Availability
   ↓
Available Slots
   ↓
Customer Selects Slot
   ↓
Confirm Booking
   ↓
Create Viewing
   ↓
Update CRM
```

Example:

```json
{
  "property_id": "A102",
  "customer_id": "C1001",
  "date": "2026-09-10",
  "time": "12:00",
  "status": "confirmed"
}
```

---

# CRM and Human Handoff

When a lead becomes qualified:

```text
AI Agent
    ↓
Lead Score
    ↓
Qualified Lead
    ↓
CRM
    ↓
Sales Representative
```

The sales representative receives the relevant context.

Example:

```text
Customer:
Ahmed

Intent:
Buy

Location:
New Cairo

Budget:
4.5M

Requirements:
3 Bedrooms

Timeline:
Within 1 Month

Recommended Properties:
A102
B205

Lead Score:
87 / Hot

Viewing:
Requested
```

The goal is to ensure the human sales representative can continue the conversation without forcing the customer to repeat their requirements.

---

# Data Sources

The system may consume data from:

- Property documents.
- Property spreadsheets.
- Property CSV files.
- Project brochures.
- Reservation contracts.
- Unit sketches.
- Floor plans.
- Property images.
- Customer information.
- Lead information.
- Interaction history.
- Existing CRM.
- Scheduling systems.
- Website interactions.
- WhatsApp conversations.

Each source should be analyzed during the architecture phase for:

- Data owner.
- Data format.
- Access method.
- Update frequency.
- Data quality.
- Required transformations.
- Security requirements.

---

# Data Model

## Property

```json
{
  "property_id": "A102",
  "project": "Project X",
  "location": "New Cairo",
  "property_type": "apartment",
  "purpose": "sale",
  "area_sqm": 145,
  "bedrooms": 3,
  "bathrooms": 2,
  "price": 4800000,
  "payment_plan": "installment",
  "availability": "available"
}
```

## Customer

```json
{
  "customer_id": "C1001",
  "name": "Ahmed",
  "language": "ar",
  "intent": "buy",
  "budget": 4500000,
  "preferred_location": "New Cairo"
}
```

## Lead

```json
{
  "lead_id": "L1001",
  "customer_id": "C1001",
  "intent": "buy",
  "budget": 4500000,
  "timeline": "within_1_month",
  "score": 87,
  "status": "hot"
}
```

## Interaction

```json
{
  "interaction_id": "I1001",
  "customer_id": "C1001",
  "channel": "whatsapp",
  "intent": "property_search",
  "message": "عايز شقة 3 غرف في التجمع",
  "timestamp": "2026-09-07T18:00:00"
}
```

---

# Project Structure

```text
ai-real-estate-assistant/
│
├── app/
│   │
│   ├── agents/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── router.py
│   │   ├── property_agent.py
│   │   └── sales_agent.py
│   │
│   ├── rag/
│   │   │
│   │   ├── ingestion/
│   │   │   ├── pipeline.py
│   │   │   ├── document_loader.py
│   │   │   ├── pdf_loader.py
│   │   │   ├── office_loader.py
│   │   │   ├── spreadsheet_loader.py
│   │   │   ├── image_loader.py
│   │   │   ├── chunker.py
│   │   │   └── metadata.py
│   │   │
│   │   ├── ocr/
│   │   │   ├── extractor.py
│   │   │   ├── document_parser.py
│   │   │   └── post_processor.py
│   │   │
│   │   ├── cv/
│   │   │   ├── image_processor.py
│   │   │   ├── vision_extractor.py
│   │   │   └── floorplan_parser.py
│   │   │
│   │   ├── normalization/
│   │   │   ├── property_normalizer.py
│   │   │   └── field_mapper.py
│   │   │
│   │   ├── embeddings.py
│   │   │
│   │   ├── retrieval/
│   │   │   ├── hybrid_search.py
│   │   │   ├── metadata_filter.py
│   │   │   └── reranker.py
│   │   │
│   │   └── generation.py
│   │
│   ├── leads/
│   │   ├── qualification.py
│   │   ├── scoring.py
│   │   └── handoff.py
│   │
│   ├── scheduling/
│   │   ├── availability.py
│   │   └── booking.py
│   │
│   ├── crm/
│   │   ├── client.py
│   │   ├── customers.py
│   │   ├── leads.py
│   │   └── interactions.py
│   │
│   ├── llm/
│   │   ├── client.py
│   │   ├── prompts.py
│   │   └── structured_output.py
│   │
│   ├── database/
│   │   ├── postgres.py
│   │   ├── vector_db.py
│   │   ├── models.py
│   │   └── repositories/
│   │       ├── property.py
│   │       ├── customer.py
│   │       └── lead.py
│   │
│   ├── channels/
│   │   ├── website.py
│   │   └── whatsapp.py
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py
│   │       ├── properties.py
│   │       ├── documents.py
│   │       └── webhooks.py
│   │
│   ├── observability/
│   │   ├── logging.py
│   │   ├── tracing.py
│   │   └── metrics.py
│   │
│   ├── schemas/
│   │   ├── property.py
│   │   ├── customer.py
│   │   ├── lead.py
│   │   └── interaction.py
│   │
│   ├── config.py
│   └── main.py
│
├── data/
│   ├── raw/
│   │   ├── properties/
│   │   ├── documents/
│   │   └── images/
│   │
│   ├── processed/
│   │   ├── text/
│   │   ├── chunks/
│   │   ├── metadata/
│   │   └── extracted/
│   │
│   └── evaluation/
│       ├── questions.json
│       ├── golden_dataset.json
│       └── expected_recommendations.json
│
├── tests/
│   ├── unit/
│   │   ├── test_ocr.py
│   │   ├── test_retrieval.py
│   │   ├── test_property_search.py
│   │   └── test_lead_scoring.py
│   │
│   └── integration/
│       ├── test_rag.py
│       ├── test_agent.py
│       ├── test_crm.py
│       └── test_scheduling.py
│
├── scripts/
│   ├── ingest.py
│   └── evaluate.py
│
├── .env
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Technology Areas

The final technology stack should be selected after analyzing the existing company infrastructure, available data, integration requirements, cost, and MVP timeline.

Expected technology areas include:

| Area | Responsibility |
|---|---|
| Python | Backend and AI implementation |
| FastAPI | API layer |
| LangGraph | Agent workflow orchestration |
| LLM | Understanding, extraction, reasoning, response generation |
| Embeddings | Semantic representation |
| Vector Database | Semantic retrieval |
| Relational Database | Structured application data |
| OCR | Text extraction |
| Computer Vision | Visual understanding |
| Reranker | Retrieval ranking |
| CRM API | Customer/lead synchronization |
| Scheduling API | Viewing availability |
| WhatsApp | Customer communication |
| Website | Customer communication |
| Docker | Deployment |
| Logging | Application observability |
| Tracing | AI workflow tracing |
| Metrics | Performance and evaluation |

Specific vendors and models should be finalized during the technical architecture phase.

---

# AI Design Principles

## 1. The LLM Does Not Own Business Truth

The LLM should not invent:

```text
Properties
Prices
Availability
Payment Plans
Lead Scores
Viewing Slots
CRM Records
```

These should come from trusted data sources or deterministic business services.

---

## 2. Separate Understanding from Execution

```text
LLM
 ↓
Understand / Extract
 ↓
Structured Data
 ↓
Business Logic
 ↓
Tool / System Action
```

Example:

```text
Customer
"عايز شقة 3 غرف في التجمع."

        ↓

LLM
Extracts requirements

        ↓

RAG
Finds matching properties

        ↓

LLM
Generates grounded response
```

---

## 3. Separate Retrieval from Generation

```text
Customer Query
      ↓
Retrieval
      ↓
Evidence
      ↓
Generation
```

The LLM should generate from retrieved evidence rather than relying on its internal knowledge for current property information.

---

## 4. Structured Filtering for Hard Constraints

Hard constraints should be handled by structured filters.

Examples:

```text
price <= 5M
bedrooms = 3
location = New Cairo
purpose = sale
availability = available
```

---

## 5. Hybrid Retrieval

The retrieval system can combine:

```text
Structured Filtering
        +
Keyword / Lexical Search
        +
Semantic Search
        ↓
Candidate Set
        ↓
Reranker
        ↓
Top-K Properties
```

---

## 6. Source Traceability

Each retrieved property or chunk should retain its source metadata.

Example:

```json
{
  "property_id": "A102",
  "source_file": "project_x_brochure.pdf",
  "page": 12,
  "chunk_id": "chunk_0012",
  "location": "New Cairo",
  "price": 4800000
}
```

This allows the system to trace recommendations back to source information.

---

## 7. Deterministic Business Operations

The following should be controlled by application logic:

- Property hard filters.
- Lead scoring.
- Viewing availability.
- Booking.
- CRM updates.
- Authentication.
- Authorization.
- Validation.

The LLM can determine **what action is needed**, but the application should control **how that action is executed**.

---

# Evaluation

The MVP has the following target metrics:

| Metric | Target |
|---|---:|
| Document extraction accuracy | ≥ 95% |
| Property recommendation accuracy | ≥ 90% |
| Intent / query understanding | ≥ 85% |
| Initial agent response latency | ≤ 3 seconds |

These targets must be measured using proper evaluation datasets.

---

# RAG Evaluation

The RAG pipeline should be evaluated independently from the final LLM answer.

Recommended evaluation metrics:

- Recall@K.
- Precision@K.
- MRR.
- Retrieval relevance.
- Reranking quality.
- Metadata filtering accuracy.
- Groundedness.
- Source traceability.
- Recommendation correctness.

Example:

```json
{
  "question": "عايز شقة 3 غرف في التجمع بحد أقصى 5 مليون",
  "expected_properties": [
    "A102",
    "B205"
  ]
}
```

---

# Document Extraction Evaluation

OCR and document extraction should be evaluated against labeled ground-truth data.

Example:

```text
Ground Truth
─────────────
Property ID: A102
Area: 145
Bedrooms: 3
Price: 4.8M

Extracted
─────────
Property ID: A102
Area: 145
Bedrooms: 3
Price: 4.8M

Result:
Correct
```

The 95% extraction target should be measured using an agreed field-level or document-level evaluation methodology.

---

# Security and Reliability

## Security

The system should include:

- Secure credential management.
- Environment-variable configuration.
- Authentication where required.
- Authorization for protected operations.
- Input validation.
- Output validation.
- Customer-data protection.
- Secure integration credentials.
- Access control.

## Reliability

The system should support:

- Timeouts.
- Error handling.
- Retries for transient failures.
- Graceful degradation.
- Validation before business actions.
- Idempotency for critical operations.
- Failure recovery.

Critical actions such as booking and CRM updates should not be executed blindly based on generated text.

---

# Observability

The system should provide visibility into the complete AI workflow.

Track:

- Request latency.
- Retrieval latency.
- LLM latency.
- Token usage where applicable.
- Retrieved documents.
- Retrieved properties.
- Reranking results.
- Agent decisions.
- Tool calls.
- CRM operations.
- Scheduling operations.
- Errors.
- Failed requests.
- Lead conversion events.

A trace should make it possible to follow:

```text
Customer Message
      ↓
Agent Decision
      ↓
Requirement Extraction
      ↓
Property Retrieval
      ↓
Retrieved Properties
      ↓
Reranking
      ↓
LLM Response
      ↓
Business Action
```

---

# MVP Constraints

The project has a fixed **8-week implementation window**.

Therefore:

- MVP scope must remain strict.
- Existing company infrastructure should be reused where possible.
- Existing company data should remain the primary source of truth.
- Existing systems should be integrated instead of replaced.
- Unnecessary microservices should be avoided.
- Architecture should remain modular.
- Critical business operations should remain deterministic.
- Arabic and English support are required.
- Advanced capabilities should be postponed.

The MVP should prioritize:

```text
Core Functionality
        ↓
Accuracy
        ↓
Reliability
        ↓
Integration
        ↓
Observability
```

rather than adding unnecessary features.

---

# 8-Week Roadmap

## Week 1 — Requirements and Project Setup

### Objectives

Understand the business problem and establish the project foundation.

### Tasks

- Collect requirements.
- Identify stakeholders.
- Define functional requirements.
- Define non-functional requirements.
- Approve requirements document.
- Create repository.
- Prepare development environment.
- Define detailed Week 1 plan.

### Deliverables

- Requirements Document.
- Stakeholder list.
- Development environment.
- Repository.
- Week 1 action plan.

---

# Week 2 — Data Analysis and Technical Architecture

### Objectives

Analyze available data and design the technical architecture.

### Tasks

- Analyze property data.
- Identify data sources.
- Analyze data quality.
- Identify supported document formats.
- Identify required APIs.
- Analyze CRM integration.
- Analyze scheduling integration.
- Select technology stack.
- Define data model.
- Design system architecture.
- Design RAG architecture.
- Approve architecture.

### Deliverables

- Data quality analysis.
- Data source inventory.
- Required API list.
- Technology stack.
- Data model.
- Architecture diagrams.
- Technical architecture document.
- Week 3 plan.

---

# Week 3 — Data Processing and RAG Foundation

### Objectives

Build the property knowledge foundation.

### Tasks

- Implement ingestion pipeline.
- Implement document type detection.
- Implement document loaders.
- Implement OCR.
- Implement Computer Vision where required.
- Extract property information.
- Normalize fields.
- Generate metadata.
- Implement chunking.
- Generate embeddings.
- Configure vector database.

### Deliverables

- Working ingestion pipeline.
- OCR pipeline.
- CV processing foundation.
- Normalized property data.
- Vector database.
- Initial property knowledge base.

---

# Week 4 — Retrieval and Recommendation

### Objectives

Build accurate property discovery.

### Tasks

- Implement metadata filtering.
- Implement semantic retrieval.
- Implement hybrid retrieval.
- Implement reranking.
- Build recommendation pipeline.
- Create retrieval evaluation dataset.
- Evaluate retrieval quality.
- Tune retrieval parameters.

### Deliverables

- Hybrid retriever.
- Reranker.
- Property recommendation pipeline.
- Retrieval evaluation dataset.
- Initial retrieval metrics.

---

# Week 5 — Conversational Agent

### Objectives

Connect the LLM with the property knowledge and business workflow.

### Tasks

- Implement conversational agent.
- Implement intent detection.
- Implement requirement extraction.
- Implement structured outputs.
- Implement conversation state.
- Integrate RAG.
- Implement grounded response generation.
- Implement clarification questions.
- Validate hallucination controls.

### Deliverables

- Conversational agent.
- Requirement extraction.
- RAG-integrated agent.
- Grounded responses.
- End-to-end customer conversation.

---

# Week 6 — Lead Qualification, Scheduling and CRM

### Objectives

Connect AI conversations with sales operations.

### Tasks

- Implement lead qualification.
- Implement deterministic lead scoring.
- Implement lead status.
- Implement human handoff.
- Implement representative availability.
- Implement viewing booking.
- Integrate CRM.
- Store interaction history.

### Deliverables

- Lead qualification pipeline.
- Lead scoring engine.
- Human handoff.
- Scheduling workflow.
- CRM integration.
- Interaction tracking.

---

# Week 7 — Channels, Dashboard and Observability

### Objectives

Connect the system to customer-facing channels and operational monitoring.

### Tasks

- Website integration.
- WhatsApp integration.
- Webhook handling.
- Logging.
- Tracing.
- Metrics.
- Agent performance monitoring.
- Operational dashboard.

### Deliverables

- Website channel.
- WhatsApp channel.
- Observability system.
- Dashboard.
- Operational metrics.

---

# Week 8 — Evaluation and Production Readiness

### Objectives

Validate the complete system and prepare the MVP for deployment.

### Tasks

- End-to-end testing.
- RAG evaluation.
- Recommendation evaluation.
- OCR evaluation.
- Document extraction evaluation.
- Intent evaluation.
- Performance testing.
- Security review.
- Reliability review.
- Deployment preparation.
- Final documentation.

### Deliverables

- Evaluation report.
- Test results.
- Security review.
- Performance results.
- Deployment-ready MVP.
- Handover documentation.

---

# Running the Project

## Clone Repository

```bash
git clone <repository-url>
cd ai-real-estate-assistant
```

## Create Virtual Environment

```bash
python -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment

```bash
cp .env.example .env
```

Configure the required credentials, API keys, database URLs, and service endpoints.

Never commit `.env` to source control.

## Run API

```bash
uvicorn app.main:app --reload
```

---

# API

Expected MVP API areas:

```text
/chat
/properties
/documents
/webhooks
```

## `/chat`

Handles customer conversations.

Responsibilities may include:

- Intent detection.
- Requirement extraction.
- RAG retrieval.
- Recommendation.
- Lead qualification.
- Business workflow orchestration.

---

## `/properties`

Handles property-related retrieval operations.

Potential operations:

- Search.
- Filtering.
- Retrieval.
- Recommendation candidates.

---

## `/documents`

Handles document ingestion and processing.

Potential operations:

- Upload.
- Document processing.
- OCR.
- CV processing.
- Extraction.
- Normalization.
- Ingestion.

---

## `/webhooks`

Handles external events.

Potential sources:

- WhatsApp.
- CRM.
- Scheduling.
- Website.

The exact API contracts should be finalized during the technical architecture and integration phases.

---

# Testing

Run:

```bash
pytest
```

## Unit Tests

```text
tests/unit/test_ocr.py
tests/unit/test_retrieval.py
tests/unit/test_property_search.py
tests/unit/test_lead_scoring.py
```

## Integration Tests

```text
tests/integration/test_rag.py
tests/integration/test_agent.py
tests/integration/test_crm.py
tests/integration/test_scheduling.py
```

## Evaluation Dataset

```text
data/evaluation/
├── questions.json
├── golden_dataset.json
└── expected_recommendations.json
```

The evaluation dataset should include:

- Customer questions.
- Expected intent.
- Expected structured requirements.
- Expected retrieved properties.
- Expected answers.
- Ground-truth document fields.
- Lead qualification cases.
- Lead scoring cases.

---

## More Languages

The system can be extended to support additional languages.

---

## Commercial and Industrial Real Estate

The same architecture can later support:

- Commercial properties.
- Industrial properties.
- Investment properties.

---

## Advanced Analytics

Future versions may include:

- Sales analytics.
- Customer behavior analysis.
- Market analytics.
- Funnel analytics.
- Agent performance analytics.

---

## Predictive AI

Future predictive capabilities may include:

- Property price prediction.
- Demand forecasting.
- Market trend prediction.
- Risk analysis.

---

## AR and Virtual Property Experiences

Future versions may support:

- Virtual property tours.
- Augmented Reality.
- Interactive floor plans.
- Immersive property exploration.

---

# Project Philosophy

The project follows a strict separation of responsibilities.

```text
                    ┌─────────────────────┐
                    │        LLM          │
                    │                     │
                    │ Understand          │
                    │ Extract             │
                    │ Reason              │
                    │ Generate             │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │        RAG          │
                    │                     │
                    │ Retrieve Knowledge  │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │ Business Logic      │
                    │                     │
                    │ Validate            │
                    │ Filter              │
                    │ Score               │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │       Tools         │
                    │                     │
                    │ Schedule             │
                    │ CRM                 │
                    │ Handoff             │
                    └─────────────────────┘
```

The central principle is:

> **The LLM understands.  
> RAG retrieves.  
> Business logic validates.  
> Tools execute.  
> CRM records.**

Therefore, this project is not simply:

```text
Chatbot + RAG
```

It is:

```text
Real Estate Documents
        ↓
Property Knowledge
        ↓
RAG
        ↓
Conversational AI
        ↓
Property Recommendation
        ↓
Lead Qualification
        ↓
Viewing Scheduling
        ↓
CRM
        ↓
Human Sales Team
```

The architecture is designed to provide a focused **8-week production-oriented MVP** while maintaining clear boundaries for future expansion.