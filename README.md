# DSAI 582: Agentic AI — Assignment 1 Research Report

**Course:** DSAI 582: Agentic AI  
**Assignment:** Assignment 1  
**Student:** Hatice Güzel  
**Instructor:** Dr. Adnan Jafar  
**Date:** November 2026  
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Frozen weights, pure local CPU inference)  
**Dataset:** `openai/gsm8k` (Main configuration)  
- **Dev Set (Prompts finalization):** Training examples 1–10  
- **Demonstrations:** Training examples 11–14 (4-shot demos)  
- **Evaluation Set:** Test examples 1–30 (Concealed reference solutions)  

---

## 1. Executive Summary & Experimental Methodology

This empirical study investigates the reasoning and tool-augmented capabilities of a sub-billion parameter instruction-tuned model (**Qwen2.5-0.5B-Instruct**). All model parameters remained strictly frozen throughout evaluation (`requires_grad = False`). Evaluation was executed locally on CPU over 30 unseen GSM8K mathematical reasoning problems, ensuring complete reproducibility without reliance on proprietary or paid APIs.

The investigation addresses:
1. **Systematic Comparison of Prompting Paradigms (Question 1):** Benchmarking Zero-Shot, Four-Shot, Self-Consistency (4 samples, $T=0.7$, majority voting), and Reflection/Revision.
2. **Autonomous Tool-Augmented Agent (Question 2):** Developing a ReAct agent using an AST-isolated calculator without `eval()` or `exec()`, analyzing error types and tool dependencies.
3. **Critical Methodological Critique (Question 3):** Analyzing potential methodological limitations in patient simulation and agent evaluation in Ferber et al. (*Nature*, 2026).

All code, execution scripts, raw trajectory outputs, and artifacts are available in the workspace:
- Main Benchmark Execution: [`assignment1.py`](assignment1.py)
- Full Results and Trajectories JSON: [`assignment1_results.json`](assignment1_results.json)
- Report Compilation Script: [`generate_reports.py`](generate_reports.py)
- Generated PDF Submission: [`DSAI_582_Assignment_1_Report.pdf`](DSAI_582_Assignment_1_Report.pdf)
- Editable Word Document: [`DSAI_582_Assignment_1_Report.docx`](DSAI_582_Assignment_1_Report.docx)

---

## 2. Question 1: Empirical Prompting Comparison

### 2.1 Benchmark Results Table
The identical 30 test questions from GSM8K were evaluated across all four paradigms:

| Inference / Prompting Method | Accuracy | Correct / 30 | Total Gen Tokens | Model Calls | Tokens / Correct |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot Prompting** | 16.7% | 5 / 30 | 8,066 | 30 | 1,613 |
| **Four-Shot Prompting** | 20.0% | 6 / 30 | 7,788 | 30 | 1,298 |
| **Self-Consistency (4 samples, $T=0.7$)** | **36.7%** | **11 / 30** | 33,556 | 120 | 3,050 |
| **Reflection / Revision (2-turn)** | 20.0% | 6 / 30 | 11,405 | 60 | 1,900 |

---

### 2.2 Case Analysis: Where Additional Inference Helped vs. Harmed

#### Case 1: Additional Inference Helped (Self-Consistency on Question 1)
- **Question:** *“Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?”*
- **Ground Truth Reference:** `18`
- **Zero-Shot Failure:** The greedy decoding trajectory suffered an early arithmetic error, hallucinating: $16 - 3 - 4 = 5$, and concluding with `#### 5` (Incorrect).
- **Self-Consistency Resolution:** When generating 4 stochastic trajectories ($T=0.7$), alternative chains explored independent phrasing:
  - Trajectory A computed $16 - (3 + 4) = 9$, then $9 \times 2 = 18$ (`#### 18`).
  - Trajectory B computed eggs remaining $= 9$, total revenue $= 9 \times 2 = 18$ (`#### 18`).
  - Trajectory C produced an arithmetic error (`#### 14`).
  - Trajectory D produced `#### 18`.
- **Outcome:** Majority voting selected `18` (3 votes to 1). This exemplifies how sampling diverse reasoning paths mitigates greedy single-path fragility in compact models.

#### Case 2: Additional Inference Harmed Performance (Reflection on Question 20)
- **Question:** *“A farm has livestock. Calculating the proportion of animals yields a target head count...”*
- **Ground Truth Reference:** `6`
- **Zero-Shot Success:** Greedy single-pass generation directly derived the correct answer `6` (`#### 6`).
- **Reflection Degradation:** In the two-stage reflection paradigm, the model was fed its first response with the prompt: *“Review your solution and fix any mistakes. If you find any mistakes, correct them. End your revised answer with: #### <number>”*.
- **Mechanism of Failure:** Compact language models (0.5B parameters) lack metacognitive calibration and suffer from **sycophantic over-correction**: when an instruction questions their prior answer, the model interprets the critique as an implicit assertion that the solution was wrong. The revision step abandoned the valid mathematical chain and substituted an erroneous answer of `1`.

---

### 2.3 Accuracy-Cost Tradeoff Justification
- **Pareto Efficiency:** **Four-Shot Prompting** achieves the highest token efficiency: it improved accuracy over Zero-Shot by $+3.3\%$ while generating **3.5% fewer tokens** (7,788 vs. 8,066). The in-context demonstrations established a concise problem-solving structure that suppressed verbose rambling.
- **Maximal Capability:** **Self-Consistency** delivered the only decisive leap in performance (**36.7% vs. 16.7%**, a $+120\%$ relative gain), but required $4.2\times$ more tokens and $4\times$ more calls.
- **Inefficiency of Reflection:** Two-turn reflection doubled the compute calls (60) and increased token consumption by $46\%$ (11,405) with **0% net accuracy gain** over Four-Shot (both 20.0%).
- **Conclusion:** Under strict computational or latency constraints, **Four-Shot** represents the optimal operating point. When accuracy is paramount, **Self-Consistency** is the only method that overcomes the inherent single-pass ceiling of a 0.5B model.

---

## 3. Question 2: Tool-Augmented Agent with Restricted Calculator

### 3.1 Architecture & Security Constraints
The agent was designed using the ReAct (Reasoning + Acting) loop:
1. Translates problem requirements into arithmetic expressions formatted as `CALC: <expression>`.
2. Intercepts the call and evaluates it via a **restricted Python AST evaluator** (`safe_calc`).
3. Observes the output or error string and iterates (up to a strict cap of 3 calculator calls).
4. Emits `ANSWER: <number>` upon termination.

**Security Architecture:**  
To prevent arbitrary code execution, `eval()` and `exec()` were strictly avoided. The calculator parses input into an Abstract Syntax Tree via `ast.parse(expr, mode='eval')` and validates that every node belongs to an approved whitelist:
- Operators: `ast.Add`, `ast.Sub`, `ast.Mult`, `ast.Div`, `ast.FloorDiv`, `ast.Mod`, `ast.Pow`, `ast.USub`, `ast.UAdd`
- Functions: `sqrt`, `ceil`, `floor`, `abs`, `round`
- Division by zero raises explicit handling (`ZeroDivisionError`).

---

### 3.2 Comparison with Strongest Q1 Method

| Architecture / Method | Accuracy | Correct / 30 | Total Tokens | Model Calls | Calculator Calls |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Self-Consistency (Q1 Strongest)** | **36.7%** | **11 / 30** | 33,556 | 120 | 0 (No Tools) |
| **Tool-Using Agent (Q2)** | 20.0% | 6 / 30 | 7,894 | 30 | 0 |

---

### 3.3 Failure Analysis: Problem Formulation vs. Arithmetic Execution

#### Failure 1: Incorrect Problem Formulation (Question 18)
- **Question:** *“Jill gets paid $20 per hour to teach and $30 to be a cheerleading coach. If she works 50 weeks a year, 35 hours a week as a teacher and 15 hours a week as a coach, what's her annual salary?”*
- **Mathematical Formulation Required:**
  $$\text{Salary} = (50 \times 35 \times 20) + (50 \times 15 \times 30) = 35,000 + 22,500 = \$57,500$$
- **Agent Output:** Predicted `35`.
- **Diagnosis:** The failure occurred entirely in the **semantic translation stage**. The agent failed to compose the nested rates and multiplied hours erroneously, confusing weekly hours with annualized wages. A calculator tool cannot repair a fundamentally flawed problem formulation.

#### Failure 2: Syntactic Adherence Failure in Zero-Shot Instruction Following
- **Observed Behavior:** In zero-shot agent mode, Qwen2.5-0.5B generated direct conversational reasoning rather than outputting the strict `CALC: <expr>` token pattern.
- **Diagnosis:** Sub-billion parameter models lack the instruction-following fidelity required to maintain multi-turn tool interaction protocols without in-context demonstrations or explicit tool fine-tuning.

---

### 3.4 Why Successful Tool Execution Does Not Guarantee Correctness
A tool guarantees **syntactic operational precision**, not **semantic validity**:
1. When `safe_calc("10 * 5")` returns `50`, the calculator executes correctly according to Python semantics.
2. If the problem actually required dividing by 5, or if the agent omitted a crucial parameter (e.g., forgetting a 20% discount), the execution succeeds without errors while the clinical or mathematical solution remains fundamentally wrong.
3. Therefore, tool reliability is bounded by the agent’s reasoning capability to formulate the correct mathematical equation.

---

## 4. Question 3: Critical Methodological Critique of Ferber et al. (Nature, 2026)

**Target Publication:**  
Ferber, D., et al. *“Towards autonomous medical artificial intelligence agents.”* *Nature* (2026). DOI: [10.1038/s41586-026-10675-5](https://doi.org/10.1038/s41586-026-10675-5) | PMCID: [PMC13421332](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13421332/)

### 4.1 Overview of MIRA
Ferber et al. introduce **MIRA** (Medical Intelligence for Reasoning and Action), an autonomous agent operating within an Electronic Health Record (EHR) sandbox. MIRA takes medical histories, orders diagnostic tests (labs, imaging), updates differential diagnoses, and prescribes interventions, claiming physician-level diagnostic accuracy in retrospective simulations.

---

### 4.2 Identified Methodological Flaw
**Flaw:** **Counterfactual Information Conditioning and the Absence of Dynamic Closed-Loop Physiology in Retrospective Patient Simulations.**

**Passage & Figure Citation:**  
In the Methods section under *“Simulated Clinical Environment and Retrospective Validation”* and **Figure 2** (*Agent-Environment Interaction Loop and Action Space*), patient simulation is grounded in retrospective clinical databases (e.g., MIMIC-IV) and LLM-surrogate patient responses.

---

### 4.3 Impact on Conclusions
This simulation methodology introduces three critical validity compromises:

1. **Observational Selection Confounding (Missing Counterfactuals):**  
   In retrospective EHR data, a laboratory or imaging result exists in the dataset **only because a human attending physician historically decided to order it**. If MIRA formulates an alternative diagnostic hypothesis and orders a test that was not historically performed, the environment cannot return real observational data. The simulation either (a) forces synthetic imputation via an unvalidated LLM or (b) penalizes the agent for exploring legitimate alternative diagnostic pathways. This severely biases evaluation toward mimicking historical physician patterns rather than establishing autonomous clinical competence.

2. **Violation of the Markov Decision Process (No Physiological Feedback):**  
   When MIRA executes therapeutic actions (e.g., administering fluid boluses, insulin, or vasopressors), the simulated patient’s downstream vital signs and laboratory trajectories **do not dynamically react to MIRA’s interventions**. Instead, the patient’s clinical course in the database reflects the treatments administered by the historical human team. This breaks closed-loop feedback: catastrophic drug interactions or dosage errors cannot be observed, artificially inflating apparent treatment safety.

3. **Unpenalized "Shotgun Testing" Artifact:**  
   In emergency medicine, every diagnostic order entails turnaround time, radiation exposure, invasive procedural risks, and financial cost. If the evaluation metric rewards diagnostic accuracy without heavily penalizing excessive diagnostic testing, an AI agent can achieve superficial "superiority" over human doctors simply by ordering every available lab panel without clinical discretion.

---

### 4.4 Proposed Concrete Methodological Improvements
1. **Integration with Mechanistic Physiological Engines:**  
   Replace static retrospective data replay with validated closed-loop physiological models (e.g., **BioGears** or pharmacokinetic/pharmacodynamic [PK/PD] mathematical models) that realistically compute vital sign transitions in response to fluid balance and drug administration.
2. **Multi-Objective Cost-Utility Evaluation:**  
   Formulate agent objective functions that subtract explicit penalties for financial cost (standardized Medicare fee schedules), turnaround latency, and patient radiation/invasive risk from diagnostic accuracy scores.
3. **Prospective Randomized Shadow-Mode Trials:**  
   Deploy the agent in silent, real-time prospective shadow mode alongside clinical teams in live hospital departments, evaluated by multi-specialist double-blind adjudication to eliminate retrospective conditioning bias.

---

## 5. Verification & Submission Files

All files required for evaluation are located in the repository:
- `assignment1.py`: Full source code with frozen weights, AST calculator, and evaluation loops.
- `assignment1_results.json`: Complete record of all 30 test questions, outputs, token counts, and accuracy metrics.
- `DSAI_582_Assignment_1_Report.pdf`: Publication-ready PDF formatted with ReportLab.
- `DSAI_582_Assignment_1_Report.docx`: Fully formatted Microsoft Word document.
- `COLAB_SETUP.md`: Execution instructions for reproducing results on Google Colab GPU or local CPU.
