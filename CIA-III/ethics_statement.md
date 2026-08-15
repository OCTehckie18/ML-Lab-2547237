# Ethics Statement — Mission Crisis × Robin Hood Army

**Registration Number:** 2547237  
**Project:** Classifying Disaster Food-Aid Requests Using Ensemble ML

---

## 1. Bias & Fairness

### Language Bias
The model relies on **English translations** of messages originally written in Haitian Creole, French, and Spanish. Translation quality varies — nuanced food requests expressed in local idioms or incomplete translations may be misclassified. This creates a systematic bias against non-English speakers who form the majority of disaster-affected populations in the training data.

### Geographic Bias
The training data comes predominantly from **Haiti (2010 earthquake)** and **Pakistan (2010 floods)**. The model may underperform on messages from other regions, particularly Indian crises where Robin Hood Army primarily operates. Vocabulary, cultural references, and communication patterns differ significantly across geographies.

### Representation Bias
Populations with less access to SMS or digital communication channels — including **rural communities, elderly individuals, and people with disabilities** — are inherently under-represented in the dataset. The model cannot help those who cannot send messages in the first place.

### Socioeconomic Bias
Messages from individuals with higher literacy levels may be better classified because they use clearer, more explicit language. Those with limited education may send shorter, more ambiguous messages that the model struggles with.

---

## 2. Privacy

- The dataset is **fully anonymised** — no personally identifiable information (names, phone numbers, addresses) is present.
- All messages are sourced from **publicly available disaster-response platforms** under a CC0 (Public Domain) license.
- If deployed at RHA, incoming messages would need to be **anonymised before processing** and stored securely.
- No biometric, health, or financial data is collected or processed.

---

## 3. False-Positive vs False-Negative Cost Analysis

| Error Type | Consequence | Severity |
|---|---|---|
| **False Positive** (non-food tagged as food) | A volunteer investigates an irrelevant message | **Low** — minor time cost |
| **False Negative** (food request missed) | A family in crisis does not receive food aid | **Severe** — direct human harm |

### Design Decision
We **prioritise recall over precision**. The cost of missing a genuine food request (people going hungry) far exceeds the cost of flagging a non-food message for human review. The model threshold can be adjusted downward to increase sensitivity at the expense of more false positives.

---

## 4. Human Oversight Requirements

1. The model must **always function as a triage assistant**, never as an autonomous gatekeeper. No message should be discarded based solely on model output.
2. All messages flagged by the model should be **reviewed by a human volunteer** before action is taken.
3. The system should display **confidence scores** alongside predictions so volunteers can prioritise uncertain cases.
4. A **feedback mechanism** should allow volunteers to correct misclassifications, enabling continuous model improvement.
5. During deployment, **model performance must be continuously monitored** with periodic retraining on new data.

---

## 5. Uncertainty & Limitations

- **Temporal drift:** The model is trained on 2010–2012 disaster data. Communication patterns, vocabulary, and crisis types have evolved significantly since then (e.g., COVID-19 created entirely new aid-request patterns).
- **Domain shift:** Messages from Indian crises (RHA's primary operating region) may use different vocabulary, languages (Hindi, Assamese, Bengali), and cultural references not represented in the training data.
- **Truncated messages:** Many messages in the dataset are incomplete or truncated, which limits the model's ability to extract meaningful features.
- **Model confidence is not calibrated:** A predicted probability of 0.8 does not necessarily mean 80% of such messages are food requests. Calibration analysis would be needed before deployment.

---

## 6. Deployment Safeguards

If this model were to be deployed at Robin Hood Army:

1. **Pilot testing** with a small chapter before wide rollout.
2. **A/B testing** against manual triage to measure real-world improvement.
3. **Kill switch** — ability to immediately disable the model if it causes harm.
4. **Regular audits** — monthly review of false negatives to identify systematic failures.
5. **Diverse training data** — augment with messages from Indian disaster contexts (Assam floods, Kerala floods, COVID-19 food drives).
6. **Multilingual support** — extend to Hindi, Assamese, and Bengali for Indian operations.

---

## 7. Acknowledgements

- **Dataset:** Figure Eight (Appen) — [Multilingual Disaster Response Messages](https://www.kaggle.com/datasets/landlord/multilingual-disaster-response-messages?resource=download), CC0 License.
- **Robin Hood Army:** [robinhoodarmy.com](https://robinhoodarmy.com) — inspiration and real-world context.
- **Tools:** scikit-learn, XGBoost, LightGBM, SHAP, imbalanced-learn.

---

*This ethics statement is part of the CIA-III submission for MCA 521-4 Machine Learning.*  
*Registration Number: 2547237*
