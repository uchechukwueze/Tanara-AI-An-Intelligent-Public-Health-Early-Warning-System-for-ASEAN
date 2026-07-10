# 🩺 HealthPulse AI 
Predict Earlier. Act Faster. Save Lives.
## An Intelligent Public Health Early-Warning System for ASEAN

![SDG 3](https://img.shields.io/badge/SDG-3%20Good%20Health%20%26%20Well--Being-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-blue)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)


## 🌍 Overview

The Association of Southeast Asian Nations (ASEAN) represents more than 660 million people across countries with different healthcare strengths, challenges, and resource capacities.

Despite improvements in healthcare delivery, many communities still face preventable health risks linked to disease burden, maternal and child mortality, nutrition challenges, and unequal healthcare readiness.

Traditional public health systems often respond after problems have already escalated.

**HealthPulse AI changes this approach.**

HealthPulse AI is an AI-powered public health intelligence platform that helps identify health vulnerabilities early, predict future risk levels, and recommend targeted interventions before health challenges become crises.


---


# 🚨 Problem Statement

Across ASEAN's 660+ million population, unequal health progress and fragmented public health data make it difficult to detect emerging risks early and deploy timely, targeted interventions.


---


# 💡 Solution

HealthPulse AI transforms multiple health indicators into actionable intelligence for decision-makers.

The system:

✔ Measures current public health vulnerability  
✔ Evaluates healthcare system readiness  
✔ Identifies countries requiring urgent intervention  
✔ Predicts future health risk levels using machine learning  
✔ Provides targeted AI-driven public health recommendations


The goal is simple:

> Move healthcare decision-making from reactive response to proactive prevention.


---


# ✨ Key Features


## 📊 Health Risk Intelligence

Creates a **Health Risk Index (0–100)** using:

- Infant mortality rate
- Under-five mortality rate
- Maternal mortality rate
- Malaria prevalence
- Tuberculosis prevalence
- Undernourishment levels


Countries are classified into:

🟢 Low Risk  
🟡 Medium Risk  
🔴 High Risk


---


## 🏥 Health System Readiness Assessment

Measures how prepared countries are to respond using:

- Government health expenditure
- DPT immunization coverage
- Measles immunization coverage
- Nurses and midwives density
- Physician density


Output:

🟢 High Readiness  
🟡 Medium Readiness  
🔴 Low Readiness


---


## 🎯 Risk–Readiness Priority Matrix

HealthPulse AI combines risk and readiness scores to classify intervention priorities:


| Risk | Readiness | Priority |
|----|----|----|
| High | Low | Emergency Priority |
| High | Medium | High Priority |
| Medium | Low | Preventive Priority |
| Low | High | Stable / Resilient |


This helps determine where resources should go first.


---


# 🤖 Machine Learning Early-Warning Model


HealthPulse AI predicts the following year's health risk level using current-year indicators.


Example:

2004 Health Indicators → Predict 2005 Risk Level

2024 Health Indicators → Predict 2025 Risk Level

Models tested:

- Logistic Regression
- Random Forest Classifier


Validation approach:

- Time-based train-test split
- Older years used for training
- Recent years reserved for testing


Final model selection prioritized:

1. High-risk recall
2. Macro F1-score
3. Accuracy


Because in public health:

> Missing a high-risk country is more costly than raising an early warning.


---


# 🧠 AI Recommendation Engine


Beyond prediction, HealthPulse AI explains **why** a country is vulnerable.


The recommendation layer:

1. Compares indicators against ASEAN reference levels
2. Detects key health weaknesses
3. Identifies priority intervention areas
4. Generates targeted recommendations


Example:

**Detected Risk**

- High child mortality
- Low healthcare workforce availability


**Recommended Actions**

- Expand child health programs
- Improve healthcare workforce distribution
- Strengthen preventive healthcare systems


---


# 📂 Dataset


The dataset contains ASEAN country-level health indicators from:

**2004–2025**


Data sources:

- World Bank Development Indicators (WDI)
- International health indicator databases


The original dataset contained historical observations from 2004–2014.

To support prototype forecasting, recent years were extended using country-level trend estimation where complete official values were unavailable.


Countries covered:

- Brunei
- Cambodia
- Indonesia
- Laos
- Malaysia
- Myanmar
- Philippines
- Singapore
- Thailand
- Vietnam


---


# ⚙️ Technical Workflow

Raw Health Data

    ↓

Data Cleaning & Processing

    ↓

Feature Engineering

    ↓

Health Risk Index Creation

    ↓

Readiness Score Development

    ↓

Risk-Priority Classification

    ↓

Machine Learning Prediction

    ↓

AI Recommendation Engine

    ↓

Streamlit Decision Dashboard




---


# 🛠️ Technologies Used


### Programming & Analysis

- Python
- Pandas
- NumPy


### Machine Learning

- Scikit-learn
- Logistic Regression
- Random Forest


### Visualization

- Plotly
- Matplotlib
- Seaborn


### Deployment

- Streamlit


### Model Persistence

- Joblib


---


# 📊 Application Pages


## 1. Executive Overview

Regional health overview showing:

- Countries covered
- Historical trends
- Current risk distribution


## 2. Risk Intelligence

Explore:

- Highest-risk countries
- Risk evolution over time
- Country comparisons


## 3. Readiness & Priority

Displays:

- Healthcare readiness
- Risk-readiness matrix
- Intervention priorities


## 4. AI Prediction Copilot

Allows users to:

- Select country and year
- Predict next-year risk
- View recommended actions


## 5. Country Deep Dive

Tracks individual country:

- Risk score
- Readiness score
- Historical progress


## 6. Methodology

Explains:

- Framework
- Model approach
- Assumptions


---


# 🌱 Sustainable Development Goals


## Primary Goal

### SDG 3: Good Health and Well-being

HealthPulse AI supports:

- Early disease risk detection
- Stronger healthcare preparedness
- Better allocation of health resources


## Secondary Goals

### SDG 9: Industry, Innovation and Infrastructure

Using AI innovation to strengthen digital health systems.


### SDG 10: Reduced Inequalities

Identifying vulnerable populations and improving resource prioritization.


---


# ❤️ Inspiration


HealthPulse AI was inspired by a personal experience within our team.

One teammate lost his father to tuberculosis, reminding us that behind every health statistic is a real person, family, and community.

We built HealthPulse AI because many health challenges should not only be treated after they happen — they should be anticipated and prevented.


---


# 🚀 Future Improvements


Future versions aim to integrate:

- Real-time health surveillance data
- Climate and environmental indicators
- Regional/subnational health data
- Automated live data pipelines
- Advanced AI recommendation models


---


# 👥 Team

Built for the **10Alytics Global Hackathon 2026**

By a team passionate about using:

**Data + Artificial Intelligence + Innovation**

to build healthier and more resilient communities.


---


# 🩺 HealthPulse AI

### Predict Earlier. Act Faster. Save Lives.



The final solution was deployed as an interactive Streamlit application called HealthPulse AI. 

The app allows users to explore health risk trends, compare health system readiness, identify emergency-priority countries, and generate next-year risk predictions. 

It also provides AI-supported public health recommendations based on each country’s risk profile and readiness gaps. 

This transforms the analysis from a static dashboard into an early-warning decision-support system for public health planning.


The final solution was deployed as an interactive Streamlit application called HealthPulse AI. 

The app allows users to explore health risk trends, compare health system readiness, identify emergency-priority countries, and generate next-year risk predictions. 

It also provides AI-supported public health recommendations based on each country’s risk profile and readiness gaps. 

This transforms the analysis from a static dashboard into an early-warning decision-support system for public health planning.
