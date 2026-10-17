# LLM Output Evaluation: A Detailed Reference

## Methods, metrics, evaluator models, and how their scores are produced

**Reference date:** September 16, 2026.  
**Scope:** Every evaluation approach named in the preceding discussion, including G-Eval, DAG evaluation, FActScore, Ragas Faithfulness and Factual Correctness, QAGS, QAFactEval, question-based coverage, DeepEval SummarizationMetric, pairwise evaluation, Prometheus 2, UniEval, PoLL, GPTScore, AlignScore, BERTScore, and deterministic checks.

The central question is not simply which evaluator is better. It is **what each evaluator actually does**, which parts are shared with other methods, and what its result means.

All conversations, rubrics, intermediate judgments, and numerical examples below are illustrative. They are not results from executing these evaluators. Descriptions of published methods and library behavior cite primary papers, official documentation, or official source code. Library behavior is version-sensitive; a moving source-code branch is not a reproducible version pin.

## Contents

- [1. Foundations: what counts as an evaluation?](#1-foundations-what-counts-as-an-evaluation)
- [2. Direct and structured LLM judging](#2-direct-and-structured-llm-judging)
- [3. Claim-based evaluation](#3-claim-based-evaluation)
- [4. Question-based evaluation](#4-question-based-evaluation)
- [5. Comparing outputs and combining judges](#5-comparing-outputs-and-combining-judges)
- [6. Specialized models, likelihoods, and embeddings](#6-specialized-models-likelihoods-and-embeddings)
- [7. Deterministic and rule-based evaluation](#7-deterministic-and-rule-based-evaluation)
- [8. How the approaches relate to one another](#8-how-the-approaches-relate-to-one-another)
- [9. Understanding scores and evaluating the evaluators](#9-understanding-scores-and-evaluating-the-evaluators)
- [10. Glossary](#10-glossary)
- [Sources](#sources)

---

## 1. Foundations: what counts as an evaluation?

### 1.1 The word "eval" hides several different things

An evaluation can be understood as a chain:

```text
Candidate output + evidence + evaluation criterion
                         |
                 Measurement procedure
                         |
                 Intermediate findings
                         |
                  Scoring/aggregation
                         |
              Score, verdict, or comparison
```

The following distinctions prevent most terminology confusion:

| Layer | Meaning | Example |
|---|---|---|
| Evaluation target | The thing being inspected. | One generated conversation summary. |
| Quality dimension | The property of interest. | Faithfulness, completeness, relevance. |
| Evidence | Material against which the output is checked. | Conversation messages, tool results, reference facts. |
| Method | The procedure used to inspect the output. | Direct scoring, claim verification, question answering. |
| Judge or scoring model | The component interpreting language. | A general LLM, Prometheus 2, an alignment classifier. |
| Metric | The defined measurement, including its scoring rule. | Supported claims divided by extracted claims. |
| Library | Software implementing and running evaluations. | DeepEval or Ragas. |
| Benchmark or test set | The cases over which evaluations run. | A collection of stored conversations with reviewed outcomes. |
| Decision policy | What the application does with the result. | Fail on a critical contradiction, regardless of an average score. |

This is a conceptual decomposition, not a universal naming standard. A research project may use "framework" to describe a method, while a software project uses the same word for a library.

For example, DeepEval implements `GEval`, `DAGMetric`, and `SummarizationMetric`; those are different metrics or workflows within one library. Ragas similarly exposes multiple measurements, including Faithfulness and Factual Correctness.[^geval-doc][^dag-doc][^summary-doc][^ragas-faith][^ragas-factual]

### 1.2 The same machinery can measure different properties

These two instructions could use the same LLM and identical calling code:

```text
Judge whether this summary makes unsupported factual claims.

Judge whether this summary includes the required outcome.
```

The first measures faithfulness; the second measures completeness. Different criteria do not necessarily imply different algorithms.

Conversely, faithfulness can be estimated through direct ratings, atomic-claim checks, questions and answers, or a trained alignment model. Different algorithms can target the same property.[^geval][^factscore][^qags][^align]

### 1.3 Running example

The examples use this fictional conversation:

```text
[1] User:
Please request a refund for order 1234. The item arrived damaged.

[2] Tool:
{
  "order_id": "1234",
  "refund_request_id": "RF-77",
  "refund_status": "pending_review",
  "payment_status": "not_issued"
}

[3] Assistant:
The refund request has been submitted. Approval is still pending,
and no refund payment has been issued.
```

Assume the summary requirements are to preserve the order, the requested action, and the final approval/payment state. The damage reason and internal request identifier are optional for this particular example.

**Candidate A: faithful and complete under those requirements**

```text
A refund was requested for order 1234; approval remains pending,
and no payment has been issued.
```

**Candidate B: incorrect outcome**

```text
A refund was requested for order 1234, approved, and paid.
```

**Candidate C: supported but incomplete**

```text
A refund was requested for order 1234.
```

Candidate C demonstrates why "everything it says is supported" and "it says everything required" are separate measurements.

### 1.4 Three forms of external information

A **source-grounded** evaluation checks the output against the material it describes. Here, that is the conversation and its tool results.

A **reference-based** evaluation compares the output with a provided answer, label, or reference summary. The reference is an intended answer, not necessarily the complete source.

An **external-knowledge** evaluation verifies claims against another knowledge source, such as a document collection. FActScore's original experiments used a reliable knowledge source for atomic facts; that differs from merely comparing against one reference answer.[^factscore]

These forms can be combined. "Reference-free" does not mean that the evaluator has no evidence. A summary can be checked against its source without having a human-written reference summary.

---

## 2. Direct and structured LLM judging

### 2.1 LLM-as-a-judge: the broad family

**What it is.** An LLM receives an evaluation task and produces an assessment of another output. It may assign a score, classify a failure, compare two candidates, or explain a verdict. LLM-as-a-judge is a broad approach, not one standardized algorithm.[^mtbench]

**Inputs.** Typically a task description, the candidate, a criterion or rubric, and any relevant source or reference.

**Mechanism.** The model interprets the evidence and evaluation instructions. Its generated assessment is parsed into the application's expected structure.

A hypothetical result for Candidate B is:

```json
{
  "criterion": "faithfulness",
  "verdict": "fail",
  "reason": "The summary changes pending approval and no payment into approval and payment.",
  "evidence_sequences": [2, 3]
}
```

This JSON is an explanation of a decision, not independent proof that the decision is correct. A judge can cite the wrong message, misunderstand a state change, or apply the wrong criterion.

**What changes between implementations.** One implementation might ask for one overall score; another might request claim-by-claim findings. Some use examples, some use reference answers, and some judge outputs in pairs. The umbrella term does not specify these details.

**Limitations.** The MT-Bench study examined position bias, preference for verbosity, self-enhancement bias, and limitations in reasoning. Those findings concern the studied judges and tasks; they are not a universal numerical error rate for all models.[^mtbench]

A useful conceptual distinction is that structured output constrains **how a judgment is represented**. It does not guarantee **that the judgment is true**.

### 2.2 G-Eval

**What it is.** G-Eval is a prompted evaluation procedure combining a task description, evaluation criteria, generated evaluation steps, and form-based rating. Its original scoring method refines a discrete rating using a probability-weighted sum of possible rating values.[^geval]

**Inputs.** A candidate, the evaluation criterion, and the evidence required by that criterion. A human-written reference answer is not inherently required.

#### The procedure

```text
Define a quality criterion
          |
Generate detailed evaluation instructions
          |
Apply those instructions to the candidate and evidence
          |
Obtain ratings and their probabilities, or estimate their distribution
          |
Calculate the weighted rating
```

For the refund example, an illustrative criterion is:

```text
Check whether factual assertions in the summary are supported by the
conversation. Preserve approval and payment status. Evaluate omissions
separately rather than treating them as unsupported claims.
```

Illustrative steps would identify the outcome assertions, locate the corresponding tool evidence, and apply a severity-anchored rubric.

#### The scoring operation

Let the allowed ratings be a set `S`. If `p(s)` is the normalized probability assigned to rating `s`, the score is:

$$
G = \sum_{s \in S} p(s)s.
$$

For hypothetical probabilities `P(3)=0.10`, `P(4)=0.60`, and `P(5)=0.30`:

```text
G = 3 * 0.10 + 4 * 0.60 + 5 * 0.30 = 4.2
```

This is an expected rating. It is not a claim that 84% of the response is correct. The paper also estimated the rating distribution through repeated sampling when direct token probabilities were unavailable; its GPT-4 experiment used 20 samples.[^geval]

#### Original method versus DeepEval's implementation

DeepEval allows explicit evaluation steps instead of generating them. It supports custom rubrics and normalized metric scores. Its documented probability-weighting path depends on usable model log probabilities; a custom-model integration without the required raw-response support can fall back to the directly generated rating.[^geval-doc]

The source implementation also contains distinct paths for generating steps, obtaining a rating, and handling score calculation. One metric invocation is therefore not a universal promise of one model request.[^geval-code]

#### What is distinctive, and what remains shared?

G-Eval structures a direct LLM judgment. The distinctive scoring operation concerns **probabilities of rating values**, not probabilities of the candidate's words. This distinction will matter when comparing it with GPTScore.

A frozen rubric can make the scoring procedure more controlled, but the model still interprets concepts such as material error and adequate relevance. More precise decimals do not remove that judgment.

### 2.3 DAG-based evaluation

**What it is.** DAG generally means directed acyclic graph. DeepEval calls its implementation a "Deep Acyclic Graph" metric. Its nodes perform tasks or judgments; edges specify subsequent steps; terminal outcomes can carry predefined scores or delegate to another metric.[^dag-doc]

**Inputs.** A candidate, relevant evidence, and an explicitly constructed evaluation graph.

#### The procedure

Instead of asking the model to balance every issue into one rating, the graph controls how findings affect the result:

```text
Does the output contradict the final approval/payment state?
    |
    +-- Yes --> Outcome: critical failure; score 0
    |
    +-- No --> Does it omit a required final-state detail?
                    |
                    +-- Yes --> Outcome: incomplete; score 0.5
                    |
                    +-- No --> Outcome: satisfies these checks; score 1
```

These values are illustrative policy choices, not built-in universal scores.

DeepEval supports task nodes, binary judgments, nonbinary judgments, and graph construction that connects their outputs. Paths can also share downstream nodes; a DAG need not be a simple tree.[^dag-guide]

#### What is actually deterministic?

If the first verdict is `Yes`, the example always maps it to zero. That mapping is deterministic.

But the first verdict itself may come from an LLM. Two runs can disagree about whether a paraphrase contradicts the source, causing different branches to execute. DeepEval's documentation explicitly distinguishes controlled score mapping from LLM-based branch decisions.[^dag-doc]

#### What does the example reveal?

Candidate B reaches the critical-failure outcome if the contradiction is correctly detected. Candidate C reaches the incomplete outcome if the omission is correctly detected. The graph does not let fluent wording offset a critical state error.

However, this particular early-exit graph does not report every possible defect. Once it terminates, later checks do not run. A graph that gathers all diagnostics before applying a final policy is a different design.

**Distinctive feature:** explicit control flow and score mapping. The graph is not a new source of semantic understanding; it organizes the judgments that another component makes.

---

## 3. Claim-based evaluation

### 3.1 FActScore

**What it is.** FActScore measures factual precision by breaking a generated text into atomic facts and determining what fraction are supported by a specified knowledge source. The original work distinguishes this measurement from its automated estimator, which uses language models and retrieval.[^factscore]

**Inputs.** Generated text and a knowledge source. The original package also uses a topic for its retrieval-oriented setup.[^factscore-code]

#### The procedure

```text
Generated response
       |
Decompose into atomic factual statements
       |
Retrieve relevant evidence for each statement
       |
Judge whether each statement is supported
       |
Count supported statements / total statements
```

For a text with `n` extracted facts:

$$
F = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[\text{fact}_i\text{ is supported}].
$$

Here, the indicator contributes `1` for support and `0` otherwise.[^factscore]

#### Worked example

For teaching purposes, decompose Candidate B into three units:

| Unit | Finding against the conversation |
|---|---|
| A refund was requested for order 1234. | Supported. |
| That refund was approved. | Contradicted by the pending-review state. |
| That refund was paid. | Contradicted by the not-issued state. |

With this particular decomposition, the support fraction is `1/3`, approximately `0.333`.

An actual decomposition could produce a different number of units. For instance, it might separate the order identifier from the requested action. Therefore the denominator is partly a consequence of the decomposition procedure, not a fixed property of the sentence.

#### Why "atomic" matters

The sentence "The refund was approved and paid" contains two status assertions. If it is treated as one indivisible claim, a partial error cannot be represented separately. Splitting makes partial support visible, but also changes the weighting of information.

Counting ten supported minor facts and one incorrect central outcome produces a high average. That is a consequence of equal-weight counting, not evidence that the central outcome is acceptable.

#### Definition versus implementation

The official FActScore package exposes a length penalty controlled by `gamma`, an unpenalized score, response/abstention statistics, and per-response fact counts. Thus a package's final reported value is not always just the simple support fraction. Those settings must be stated when interpreting a result.[^factscore-code]

A conversation-grounded adaptation can use the transcript directly instead of retrieving encyclopedic evidence. It then measures fidelity to that conversation; it is not automatically the original biography-evaluation pipeline.

**Limit:** factual precision alone does not measure required-information coverage. Under the unpenalized support definition, Candidate C can score perfectly while omitting the outcome.

### 3.2 Ragas Faithfulness

**What it is.** Ragas Faithfulness checks whether claims in a response are supported by the supplied retrieved context. It extracts claims, checks whether the context entails them, and calculates the supported fraction on a `0-1` scale.[^ragas-faith]

**Inputs.** The question or user input, response, and retrieved contexts in the documented interface. The context is the evidence boundary; it is not a reference answer.[^ragas-faith]

#### The procedure

```text
Response + question
        |
Extract self-contained claims
        |
Check claims against the supplied contexts
        |
Supported claims / extracted claims
```

If the conversation is supplied as the evidence and the same three-unit decomposition is used, Candidate B again yields `1/3`. This resemblance to FActScore is real: both belong to the claim-support family.

The important difference is the surrounding setup. In ordinary Ragas Faithfulness, context is supplied to the metric. FActScore's original automated pipeline retrieves evidence from a knowledge source.[^ragas-faith][^factscore-code]

#### Entailment is stronger than keyword overlap

For the following pair:

```text
Context: The refund is awaiting approval.
Claim:   The refund was approved.
```

Both texts contain related words, but the claim does not follow from the context. Faithfulness asks about support, not vocabulary similarity.

Similarly:

```text
Context: The user says that a payment was issued.
Claim:   A payment was issued.
```

The context supports the user's statement as an attributed claim. Whether it supports the underlying payment event depends on the evidence policy. Attribution must not silently become independent verification.

#### Implementation details

Ragas documents an alternative in which a hallucination-detection classifier performs the claim-verification stage. A model-based metric therefore need not use a generative chat judge for every stage.[^ragas-faith]

The inspected legacy implementation returns an undefined numeric result when no statements can be extracted, rather than treating an empty response as perfectly faithful. This is an implementation-specific edge case, not an inevitable mathematical convention.[^ragas-faith-code]

**Limit:** complete faithfulness to an incomplete context can still omit required information. Also, if the context itself is wrong, consistency with it does not establish external-world truth.

### 3.3 Ragas Factual Correctness

**What it is.** This metric compares the claims in a response with a reference and reports factual precision, recall, or F1. Claim decomposition and natural-language inference turn textual facts into the counts used by those formulas.[^ragas-factual][^ragas-factual-code]

**Inputs.** The candidate response and a reference answer.

#### The scoring idea

The documented counts are:

```text
TP: Response claims supported by the reference.
FP: Response claims not supported by the reference.
FN: Reference claims missing from the response.
```

Then:

$$
P = \frac{TP}{TP+FP},\qquad
R = \frac{TP}{TP+FN},\qquad
F_1 = \frac{2PR}{P+R}.
$$

The implementation verifies response claims against the reference and reference claims against the response. This is not exact string matching or necessarily a one-to-one alignment of identical claim sets.[^ragas-factual-code]

#### Worked example

Suppose the reference represents exactly three required units:

```text
1. A refund was requested for order 1234.
2. Approval is pending.
3. No payment has been issued.
```

Candidate C contains only the first unit. With this aligned decomposition:

```text
TP = 1; FP = 0; FN = 2
Precision = 1 / 1 = 1.0
Recall    = 1 / 3 = 0.333...
F1        = 0.5
```

This makes its factual precision and incompleteness visible at the same time.

#### Decomposition settings

Ragas exposes **atomicity**, which controls how finely statements are split, and **coverage**, which controls how much detail decomposition retains. The decomposition parameter named `coverage` is not itself the final completeness score.[^ragas-factual]

#### Reference mismatch is not necessarily hallucination

Suppose a candidate additionally says the item arrived damaged. The conversation supports that claim, but the abbreviated reference above omits it.

A reference-only comparison can treat the additional claim as unsupported by the reference. That result should not automatically be reinterpreted as a contradiction of the conversation. The evaluator has compared against a different evidence boundary.

**Distinctive feature:** reference-oriented factual overlap, including omissions. The quality of the reference and the decomposition policy are part of the meaning of the score.

---

## 4. Question-based evaluation

### 4.1 QAGS

**What it is.** QAGS evaluates factual consistency by generating questions about a summary, answering those questions from both the summary and its source, and comparing the two answers. The original implementation uses question-generation and extractive question-answering models, not a general chat judge assigning a quality rating.[^qags]

**Inputs.** The source document and candidate summary; no reference summary is required.

#### The procedure

```text
Candidate summary
       |
Select answer candidates and generate questions
       |
Filter unsuitable questions
       |
For each question:
    answer from source    |    answer from summary
                          |
                 Compare the answers
                          |
                Average answer similarity
```

The original QAGS implementation compares answers with token-level F1. Its question filtering includes checks intended to remove low-quality or inconsistent questions.[^qags][^qags-code]

A simplified expression is:

$$
QAGS(X,Y)=\frac{1}{|Q|}\sum_{q\in Q}
F_1\bigl(QA(X,q),QA(Y,q)\bigr),
$$

where `X` is the source, `Y` the summary, and `Q` the retained question set.

#### Worked example

Consider these manually constructed questions for Candidate B:

| Question | Answer supported by source | Answer supported by candidate |
|---|---|---|
| Which order is involved? | 1234. | 1234. |
| What is the approval state? | Pending review. | Approved. |
| What is the payment state? | Not issued. | Paid. |

The last two disagreements provide evidence of inconsistency. These are illustrative questions, not a prediction of what QAGS's original answer-selection model would generate.

#### What is being measured?

The score measures agreement between answers to the retained questions. It does not directly count every factual error in the original summary. A defect is visible only if the question set exposes it and the answerer handles it correctly.

If Candidate C generates only a question about the order, all tested answers may agree despite its missing final status. This follows from generating questions from the candidate's existing content rather than from an independent list of required information.

#### Failure modes

Question generation might ask the wrong question. The answerer might retrieve the wrong span. Token F1 might penalize equivalent paraphrases or give partial credit to overlapping but incompatible phrases. QAGS explicitly discusses answer-similarity limitations.[^qags-code]

**Distinctive feature:** factual consistency is transformed into answer agreement, with inspectable question-answer pairs between the source and candidate.

### 4.2 QAFactEval

**What it is.** QAFactEval develops the QA-based factual-consistency approach by examining and improving its constituent components: answer selection, question generation, question answering, answer comparison, and filtering. It is a pipeline, not merely a different rubric for a chat model.[^qafacteval]

**Inputs.** A source document and generated summary.

#### The procedure

```text
Select answer spans from the summary
                  |
Generate questions targeting those spans
                  |
Filter questions not properly answerable from the summary
                  |
Answer retained questions from the source
                  |
Apply source-answerability handling
                  |
Score answer compatibility and aggregate
```

The published method includes a learned answer-comparison component, LERC, with a QuIP-initialized variant, rather than relying only on exact match or token F1. It also applies an answerability penalty when a question cannot be answered from the source. A separate filtering step removes unsuitable questions based on the summary.[^qafacteval]

The official implementation can return question-answer pairs alongside scores and provides the models composing this workflow.[^qafacteval-code]

#### Why answerability is a separate operation

Imagine asking:

```text
On what date was the refund paid?
```

The source says no payment was issued. An answerer forced to return some span might select an unrelated date from the conversation. Comparing that invented answer with the candidate's date would be misleading.

There are two different situations:

```text
Question is malformed or does not fit the summary:
    It may be inappropriate evidence and should be filtered by the method.

Question is valid for a summary claim but unsupported by the source:
    It may expose a factual error and should not simply disappear.
```

That distinction is more substantive than changing "rate accuracy" to "rate faithfulness."

#### Score interpretation

The final metric aggregates answer-comparison scores after filtering and answerability handling. Its raw values inherit the scale of the selected answer-comparison component. They should not automatically be interpreted as a fraction of supported claims or a probability of correctness.[^qafacteval-code]

Its paper also studies a combined QA-and-inference variant. That variant adds another signal; it is not evidence that all QA evaluations inherently contain an entailment model.[^qafacteval]

**Limit:** an unasked question can still hide an error, and an incorrect answerability judgment can remove or penalize the wrong evidence.

### 4.3 Required-information and question-based coverage

**What it is.** Coverage evaluation asks whether the output preserves a specified set of information requirements. A question checklist is one way to express those requirements. This is a general evaluation pattern; it is not one uniquely defined research algorithm.

For the running example, define three requirements:

```text
R1: Identify the refund request and order.
R2: Preserve the fact that approval is pending.
R3: Preserve the fact that payment has not been issued.
```

A conceptual coverage score is:

$$
Coverage=\frac{\text{requirements correctly represented}}
{\text{applicable requirements}}.
$$

Candidate C covers one of three, so this illustrative score is `1/3`.

The word **correctly** matters. Merely mentioning payment is not enough when the output gives the wrong payment state.

#### Where do requirements come from?

They can be written by a reviewer, derived from structured expected facts, or generated from the source by a model. These alternatives change the evaluation's dependency: fixed requirements depend on the author; generated requirements also depend on extraction quality.

A candidate-derived question list cannot, by itself, guarantee coverage of information absent from the candidate. To expose omissions, the evaluation needs some independent specification of what should be present.

Completeness is also relative to a target. A one-line summary should not automatically be penalized for omitting every nonessential detail in a long conversation. In the running example, the damage reason and `RF-77` are intentionally outside the required set.

This general pattern appears in DeepEval's summarization metric, which allows predefined assessment questions or generates them from the source.[^summary-doc]

### 4.4 DeepEval SummarizationMetric

**What it is.** A composite metric that combines factual alignment and information coverage. DeepEval defines the overall score as the lower of the two components:[^summary-doc]

$$
S=\min(Alignment,Coverage).
$$

**Inputs.** The original text as `input`, the summary as `actual_output`, and optionally a list of assessment questions.

#### Alignment component

The workflow extracts source information and summary claims, then judges alignment. In the inspected implementation, alignment counts positive verdicts divided by all alignment verdicts; `no` and `idk` do not receive positive credit.[^summary-code]

Conceptually, this asks:

```text
Does the summary introduce unsupported or contradictory information?
```

The component named "alignment" here is not the separate research metric **AlignScore**, discussed later.

#### Coverage component

The implementation answers assessment questions against the source and summary. Coverage counts cases where both answers are `yes`, divided by cases where the source answer is `yes`.[^summary-code]

$$
Coverage=\frac{\#(source=yes\ \land\ summary=yes)}{\#(source=yes)}.
$$

This is more specific than counting every matching yes/no answer. Matching `no` answers are not added to the positive-coverage numerator in that implementation.

To test preservation of a negative event, an appropriate illustrative question is:

```text
Does the text indicate that no refund payment has been issued?
```

Both the source and Candidate A can answer `yes`, even though the underlying event is a nonpayment. This avoids confusing the polarity of an event with the polarity of a coverage question.

#### Worked example

Suppose Candidate C receives full alignment because its only claim is supported, and it covers one of the three required facts:

```text
Alignment = 1.0
Coverage  = 1/3
Overall   = min(1.0, 1/3) = 1/3
```

The minimum implements a bottleneck: excellent alignment cannot compensate for low coverage, and excellent coverage cannot compensate for low alignment.

The workflow performs multiple model-dependent operations, including extraction and question answering. It is not equivalent to one general prompt saying "score this summary."[^summary-code]

**Limit:** the final score inherits errors from both components. Its minimum rule is an aggregation policy, not a guarantee that either component is correct.

---

## 5. Comparing outputs and combining judges

### 5.1 Pairwise evaluation

**What it is.** An evaluator receives two outputs for the same task and determines their relative quality under specified criteria. Depending on the protocol, results can include a preference, a tie, or a graded preference. The judge can be human or model-based.[^mtbench][^poll]

**Inputs.** The shared task and evidence, Candidate A, Candidate B, and comparison criteria.

#### The procedure

```text
Same task + same evidence
           |
      Output A and Output B
           |
   Apply comparison criteria
           |
   A preferred / B preferred / tie
```

An illustrative result could be:

```json
{
  "preference": "A",
  "criterion": "factual consistency",
  "reason": "A preserves pending approval and no payment; B states that approval and payment occurred."
}
```

This is a relative result. It does not assign a universally meaningful quality score to either candidate.

#### Relative preference is not absolute acceptability

Imagine comparing two incomplete summaries. One retains the order identifier and the other does not. The first may be preferable while both fail the requirement to preserve the final outcome.

Thus:

```text
A is better than B
```

does not imply:

```text
A satisfies the task requirements
```

#### Dataset-level scoring

An illustrative tie-adjusted win rate is:

$$
WinRate_A=\frac{Wins_A+0.5\,Ties}{N}.
$$

This is a chosen reporting convention. Ties can instead be reported separately or handled differently. A win rate is conditional on the opponent, test cases, criteria, and judge.

Comparing every pair of `m` systems requires `m(m-1)/2` unordered system pairs per case. Comparing each system only with a fixed baseline is a different experiment with fewer comparisons.

#### Bias and consistency

The MT-Bench study found that answer order could affect a judge's preference. Presenting both orders allows an experiment to measure such inconsistencies, though it does not guarantee that they disappear.[^mtbench]

**Distinctive feature:** the measurement is comparative. Pairwise evaluation can use a general LLM, Prometheus 2, or a panel; those are choices about the judge, not different definitions of pairwise comparison.

### 5.2 PoLL: Panel of LLM Evaluators

**What it is.** PoLL uses multiple evaluator models from different model families and pools their independently produced judgments. The paper examines several evaluation settings and aggregation choices.[^poll]

**Inputs.** The material needed for the underlying evaluation, a set of judges, and an aggregation rule.

#### The procedure

```text
                     Same evaluation case
                      /       |       \
                 Judge 1   Judge 2   Judge 3
                      \       |       /
                      Aggregation rule
                             |
                        Panel result
```

The panel can perform direct assessment, reference-based checking, or pairwise evaluation. Its defining change is who judges and how their results are combined, rather than a new definition of faithfulness.[^poll]

#### Worked example

For a hypothetical binary factuality check:

```text
Judge 1: fail
Judge 2: fail
Judge 3: pass
```

A majority-vote policy returns `fail`. A unanimity requirement would report lack of agreement instead. Both use the same underlying judgments; the aggregation policy changes the result.

For numeric ratings `2`, `2`, and `5`, the mean is `3` and the median is `2`. Neither aggregation reveals by itself which judge interpreted the evidence correctly.

These examples demonstrate general panel policies. They are not claims that the PoLL paper uses every policy shown here.

#### Independence and disagreement

Three outputs from one model can measure some within-model variability. Three distinct model families provide a different form of diversity. One model instructed to role-play three reviewers does not constitute three separately instantiated evaluator models.

However, diversity is not a proof of independent errors. Judges can share training patterns, accept the same misleading reference, or all miss the same temporal distinction. Unanimous agreement is still agreement, not external verification.

PoLL reports improvements for its particular tested models and datasets. That is empirical evidence about those settings, not a universal guarantee that a panel always beats a single judge.[^poll]

**Distinctive feature:** an ensemble of judges. It can be combined with many of the other evaluation procedures in this reference.

---

## 6. Specialized models, likelihoods, and embeddings

### 6.1 Prometheus 2

**What it is.** Prometheus 2 is a family of language models trained specifically to evaluate other model outputs. It supports rubric-based direct assessment and pairwise ranking. It is an evaluator model family, not a single scoring formula.[^prometheus]

**Inputs.** An instruction, one or two responses, evaluation criteria or a score rubric, and optionally a reference answer in the official evaluation interface.[^prometheus-code]

#### The mechanism

A general-purpose model can be prompted to evaluate. Prometheus 2 additionally learns evaluation behavior through training on assessment data. The research combines capabilities associated with direct scoring and pairwise assessment.[^prometheus]

At inference time:

```text
Task + candidate(s) + rubric + optional reference
                         |
                Trained evaluator model
                         |
                Feedback and assessment
```

The official interface's absolute-grading format returns a rating from `1` to `5`. Its relative-grading format returns `A` or `B`. A tie option is not automatically part of that standard relative-grading format.[^prometheus-code]

#### Worked example

A rubric might define an outcome contradiction as score `1`, omissions without contradiction as an intermediate score, and faithful coverage as score `5`.

Prometheus 2 could then assess Candidate B against that rubric. The essential operation still resembles direct LLM judging: a model reads instructions and produces an assessment. The distinction lies in the model's training, rather than a new mathematical aggregation of claim counts.

#### Relationship to other methods

The following describe separate layers:

```text
Rubric: Preserve the final refund state.
Procedure: Direct assessment.
Judge: Prometheus 2.
Result format: Feedback plus a 1-5 rating.
```

Changing the judge to another model does not automatically change the task. Conversely, using Prometheus 2 inside a custom claim-verification workflow changes the procedure while retaining the same model family.

Calling Prometheus 2 does not automatically reproduce G-Eval's generated-step and probability-weighting procedure. It is possible to design integrations that combine these ideas, but the integration must actually implement those operations.

**Limit:** specialized training does not make the model an oracle. A rating can still reflect an unsuitable rubric, an incomplete reference, or a mistaken interpretation. Performance has to be understood relative to the evaluated domain and task rather than inferred solely from the model's purpose.

### 6.2 UniEval

**What it is.** UniEval is a trained multidimensional evaluator that expresses evaluation dimensions as Boolean question-answering tasks. Its original approach uses a T5-based evaluator with intermediate learning and evaluation-task training.[^unieval]

**Inputs.** A candidate and a dimension-specific question, plus the source, reference, or additional context required by that dimension.

#### The mechanism

Instead of asking for an arbitrary numeric grade, UniEval asks a quality question such as:

```text
Is the generated summary consistent with the source?
```

It calculates a score from the probabilities assigned to affirmative and negative answers:

$$
s=\frac{P(Yes\mid input)}{P(Yes\mid input)+P(No\mid input)}.
$$

This is the basic scoring form. Some dimensions add sentence-level aggregation.[^unieval]

For hypothetical values `P(Yes)=0.72` and `P(No)=0.18`, the normalized score is `0.8`. This is a model preference between two answer labels, not a count of eight correct claims out of ten.

#### Why this is not QAGS

QAGS asks factual content questions and compares answers:

```text
What is the refund status?
Source answer versus summary answer.
```

UniEval asks a quality question about the texts:

```text
Is this summary consistent with the source?
Probability of Yes versus No.
```

Both use the language of question answering, but their questions serve different roles.

#### Evidence and aggregation vary by dimension

The official summarization implementation uses source text for coherence and consistency, the output alone for fluency, and a reference summary for relevance. It is therefore inaccurate to describe every UniEval dimension as reference-free.[^unieval-code]

The original method averages sentence-level scores for some dimensions. Dialogue engagingness uses a sum, so that particular aggregate is not restricted to `0-1`, even though the underlying Boolean scores are.[^unieval][^unieval-code]

#### Worked interpretation

Candidate B could be fluent but inconsistent. UniEval can express this through separate dimension scores rather than forcing one label to represent both qualities.

**Limit:** the Yes/No probability is not automatically calibrated to the real probability that a human reviewer will agree. Different dimension-specific inputs and aggregations also mean that scores should not be compared without understanding their definitions.

**Distinctive feature:** a trained evaluator with a unified Boolean-QA scoring interface, rather than open-ended numeric grading.

### 6.3 GPTScore

**What it is.** GPTScore evaluates a candidate using its conditional likelihood under a generative language model. The task description, quality aspect, and context are encoded in the prompt. The model's probabilities for the candidate text supply the score.[^gptscore]

**Inputs.** The candidate text, a task/quality description, optional source or reference material, and a model capable of scoring the candidate's tokens.

#### The mechanism

Suppose the evaluation prompt describes generating a factually faithful summary. Instead of asking the model to emit a rating, the evaluator measures how likely that model considers the already-written candidate as a continuation under the prompt.

The paper's form is:

$$
GPTScore(h)=\sum_{t=1}^{m} w_t\log P(h_t\mid h_{<t},T,\theta),
$$

where `h` is the candidate, `T` is the evaluation-conditioned prompt, and the weights specify how token log probabilities are combined.[^gptscore]

The official repository implements scoring with generative models; this is not equivalent to asking a chat model to produce a number named "GPTScore."[^gptscore-code]

#### Worked arithmetic

For an illustrative three-token sequence, suppose the token probabilities are:

```text
0.50, 0.25, 0.50
```

Using equal weights of `1/3` gives average natural-log likelihood:

```text
[log(0.50) + log(0.25) + log(0.50)] / 3 = approximately -0.924
```

This example uses a length-normalized configuration to illustrate the calculation. It does not claim that every implementation uses that exact normalization.

A less negative average means higher conditional likelihood under the same configuration. It does not mean a higher measured fraction of true statements.

#### The critical distinction from G-Eval

```text
G-Eval:
    How probable are the possible RATING VALUES?

GPTScore:
    How probable are the CANDIDATE'S TOKENS under the evaluation prompt?
```

GPTScore does not inherently produce a list of factual defects. Its observable signal is likelihood, not a generated audit report.

#### Limits implied by the measurement

A plausible or familiar sentence is not necessarily true. For the running example, a model may find a conventional refund-completion sentence linguistically likely even when the source says the payment has not occurred.

Comparisons also depend on the model, prompt, tokenizer, conditioning direction, and length normalization. Scoring candidate tokens requires access to their probabilities; probabilities only for some different, newly generated response would not compute the same quantity.

**Distinctive feature:** evaluation through conditional generation likelihood, without requiring an explicit rating judgment.

### 6.4 AlignScore

**What it is.** AlignScore evaluates factual consistency with a trained text-to-text information-alignment function. The original work trains that function using multiple task types, including inference, question answering, and fact verification.[^align]

**Inputs.** A context and a candidate claim text, which can contain multiple sentences.

#### The mechanism

For its chunk-and-sentence procedure, the context is divided into chunks and the candidate into sentences. The alignment model scores context-chunk/candidate-sentence pairs. The method selects the highest context support for each candidate sentence, then averages those maxima.[^align][^align-code]

$$
AlignScore(C,Y)=\frac{1}{n}\sum_{j=1}^{n}\max_i Align(C_i,Y_j).
$$

The official implementation provides trained RoBERTa-based checkpoints and alternative scoring modes; the selected mode is part of the metric configuration.[^align-code]

#### Worked example

Rewrite Candidate B as three sentences for this illustration:

```text
The user requested a refund for order 1234.
The refund was approved.
The refund was paid.
```

Suppose the best support scores across context chunks are:

```text
Request sentence:   0.95
Approval sentence:  0.10
Payment sentence:   0.05
```

Their average is approximately `0.367`. These are invented model scores used only to demonstrate the aggregation.

#### How it differs from atomic-fact scoring

A sentence is not necessarily an atomic fact. "The refund was approved and paid" combines two claims, but may be evaluated as one sentence-level unit.

AlignScore also aggregates learned alignment scores rather than only binary supported/unsupported labels. A value of `0.367` therefore need not mean exactly 36.7% of the facts were verified.

#### Consequences of the maximum operation

Taking the best-supporting chunk gives each sentence an opportunity to find relevant evidence without requiring the entire source to fit into one model input.

But consider a conversation with an earlier tentative approval and a later authoritative cancellation. A procedure that accepts whichever chunk most strongly supports "approved" could overlook the final state unless chronology and evidence priority are represented in the evaluation input or logic. This is a potential limitation implied by the aggregation, not a reported benchmark result for this fictional case.

**Distinctive feature:** learned information alignment with explicit chunk/sentence aggregation, rather than a generative judge producing a rubric rating.

### 6.5 BERTScore

**What it is.** BERTScore measures candidate-reference similarity through contextual token embeddings. It matches each token to the most similar token in the other text and computes precision-like, recall-like, and F1-like scores.[^bertscore]

**Inputs.** A candidate text, a reference text, and an embedding-model configuration.

#### The mechanism

```text
Candidate tokens                 Reference tokens
       |                                |
Contextual embeddings           Contextual embeddings
       \                                /
              Similarity matrix
                      |
       Best token matches in each direction
                      |
          Precision, recall, and F1
```

For normalized candidate embeddings `c_i` and reference embeddings `r_j`, the unweighted forms are:

$$
P_B=\frac{1}{m}\sum_{i=1}^{m}\max_j\cos(c_i,r_j),
$$

$$
R_B=\frac{1}{n}\sum_{j=1}^{n}\max_i\cos(r_j,c_i),\qquad
F_B=\frac{2P_BR_B}{P_B+R_B}.
$$

Matching is greedy: each token takes its best available match independently, not through a required one-to-one assignment.[^bertscore]

#### Worked interpretation

These texts can express similar information despite different wording:

```text
Reference: Approval of the refund remains pending.
Candidate: The requested refund is still awaiting approval.
```

Contextual matching can recognize similarities that exact string equality cannot. There is no requirement that a model generate a textual evaluation or numeric rating.

#### Why similarity is not factual verification

Now compare:

```text
Reference: The refund was not paid.
Candidate: The refund was paid.
```

The texts share most words but differ in the crucial negation. Contextual embeddings may reflect that difference, but the score's definition does not enforce a logical rule that the contradiction must cause failure.

Likewise, changing an order identifier can be operationally serious while affecting only a small part of a token-based average.

These examples illustrate the distinction between similarity and support; they do not assert a measured BERTScore for either pair.

#### Configuration and score meaning

The official package supports options including inverse-document-frequency weighting and baseline rescaling. The selected model and representation layer also affect the result. Consequently, a reported number is incomplete without its configuration.[^bertscore-code]

Its "precision" and "recall" are based on embedding matches, not counts of verified factual claims. They are not interchangeable with the factual precision and recall in Ragas Factual Correctness.

**Distinctive feature:** numerical similarity over contextual embeddings. It uses a learned model, but not an LLM-as-a-judge conversation.

---

## 7. Deterministic and rule-based evaluation

### 7.1 What it is

A deterministic check directly computes whether an output meets a precisely defined condition. Examples include parsing JSON, comparing identifiers, checking an allowed label, counting words under a specified counting rule, and verifying that a referenced sequence exists.

JSON Schema, for example, can express required object properties and whether additional properties are permitted. These are contract checks, not semantic judgments about whether a summary is accurate.[^json-schema]

### 7.2 The mechanism

```text
Candidate + explicit contract or expected value
                       |
                Executable check
                       |
             Pass/fail or exact count
```

Illustrative checks for a structured analyzer output are:

```text
conversation_id equals the input conversation_id
summary is an array
intent belongs to the configured label set
short_summary contains no more than the allowed number of words
source_sequence exists in the supplied conversation
```

The scoring mechanism does not depend on whether the candidate was written by a model or a human.

### 7.3 Deterministic checks can evaluate semantics when the expectation is structured

Suppose the known expected final status is the enum `pending_review` and the analyzer emits `approved`. Equality detects the error without another language model.

Similarly, a predicted intent can be compared against an adjudicated expected intent. Across cases, classification accuracy or a confusion matrix then describes agreement with those labels.

The limitation is not that code can never evaluate meaning. It is that a computable semantic check requires an appropriate representation and specification of the expected relationship.

### 7.4 What a passing rule does not prove

Checking that `source_sequence=2` exists does not prove that message 2 supports the attached claim. Checking that a quote is an exact substring proves textual presence, not that the quote establishes the claimed conclusion.

A text-based rule can also be specified incorrectly. Counting spaces is not the same as counting words under every possible contract; accepting only an exact reference sentence would reject valid paraphrases.

**Distinctive feature:** explicit computation rather than learned interpretation at evaluation time. Its reliability depends on the correctness and scope of the rule and expected data.

---

## 8. How the approaches relate to one another

### 8.1 A map of the actual operations

This table summarizes mechanisms, not relative quality. Details and source references appear in the corresponding sections.

| Name | Main operation | What its result represents |
|---|---|---|
| LLM-as-a-judge | Prompt a model to assess an output. | A judgment under supplied instructions. |
| G-Eval | Structured rating with probability-weighted rating aggregation where available. | An expected rubric rating. |
| DAG evaluation | Execute explicitly connected tasks, decisions, and terminal outcomes. | The result of the configured decision path. |
| FActScore | Decompose facts and verify support against a knowledge source. | Factual precision, with implementation-specific reporting or penalties. |
| Ragas Faithfulness | Verify response claims against supplied contexts. | The fraction of extracted claims judged supported. |
| Ragas Factual Correctness | Compare response/reference claims through inference. | Reference-based factual precision, recall, or F1. |
| QAGS | Compare answers to summary-generated questions. | Mean answer agreement for the retained questions. |
| QAFactEval | QA consistency with improved component choices and answerability handling. | Aggregated answer-comparison scores. |
| Question-based coverage | Check independently specified information requirements. | The proportion of applicable requirements correctly represented. |
| DeepEval SummarizationMetric | Combine claim alignment and question coverage. | The lower of the two component scores. |
| Pairwise evaluation | Compare two outputs under shared criteria. | Relative preference, potentially including a tie. |
| PoLL | Pool independent judgments from multiple model families. | An aggregate panel judgment. |
| Prometheus 2 | Use a language model trained for evaluation. | A rubric rating or pairwise judgment. |
| UniEval | Score Boolean questions about quality using a trained evaluator. | Dimension-specific affirmative-answer scores and aggregations. |
| GPTScore | Score the candidate's conditional token likelihood. | A model-likelihood-based quality proxy. |
| AlignScore | Score context-chunk/sentence alignment and aggregate. | Learned factual-consistency alignment. |
| BERTScore | Match contextual token embeddings. | Candidate-reference similarity. |
| Deterministic checks | Compute explicit predicates or comparisons. | Satisfaction of the specified computable conditions. |

### 8.2 Which distinctions are genuinely different?

**A changed criterion** changes the question. "Faithfulness" and "relevance" can use the same direct judge with different instructions.

**A changed decomposition** changes the unit being judged. A whole-response rating, a sentence-alignment score, and atomic-claim verification inspect different units.

**A changed evidence boundary** changes what support means. Comparing with the full conversation is not the same as comparing with an abbreviated reference.

**A changed scoring function** changes how findings are summarized. Counting, averaging, taking a minimum, or following a severity gate can produce different final values from the same findings.

**A changed model or ensemble** changes who produces the judgments. Prometheus 2 and PoLL primarily affect this part of the system.

**A changed computational mechanism** goes further. Candidate-token likelihood and embedding similarity do not require a model to generate an explicit quality verdict.

These differences can be combined rather than treated as mutually exclusive products. For example, a panel could make pairwise judgments. A DAG could contain a direct rating node. A claim-verification workflow could use either a generative judge or a trained classifier.[^poll][^dag-doc][^ragas-faith]

### 8.3 Why many methods still feel similar

Several workflows reduce to:

```text
Interpret language with a learned model
                  |
Make one or more judgments
                  |
Convert the judgments into a result
```

In those cases, much of the differentiation concerns the structure around model judgment. That structure changes the measurement and its failure modes, but does not create an independent source of truth.

An atomic-claim method does not eliminate the judgment "is this supported?" It makes that judgment repeatedly over smaller units. A DAG does not eliminate interpretation; it controls what happens after a verdict. A panel does not eliminate uncertainty; it combines several judgments.

Other methods, especially GPTScore and BERTScore, are not merely new prompts for the same numeric judge. They compute a different observable quantity.

### 8.4 Multiple dimensions are not multiple independent judges

A single call returning faithfulness, completeness, and relevance produces three dimensions from one judge invocation. It does not become a panel merely because the response has three fields.

Similarly, a prompt that says "extract claims, verify them, and report coverage" describes a combined workflow. It does not automatically reproduce the original FActScore retrieval system, QAGS answer-comparison pipeline, or G-Eval probability weighting.

Names should describe the operations actually performed, not just ideas mentioned in the prompt.

---

## 9. Understanding scores and evaluating the evaluators

### 9.1 Identical numeric ranges do not imply identical meanings

Consider four illustrative results:

```text
0.8 from a claim-support metric
0.8 from a normalized rubric rating
0.8 from a Boolean-answer scoring model
0.8 from an embedding-similarity metric
```

They can respectively mean a fraction of claims judged supported, a rescaled rating, a normalized preference for an answer label, and a similarity aggregate. Treating them as four measurements of "80% correct" erases their definitions.

Even normalization itself needs specification. Mapping rating `4` from a `1-5` scale with `(s-1)/4` produces `0.75`. Dividing by `5` produces `0.8`. These are different transformations with different minimum values.

### 9.2 Exact arithmetic does not guarantee accurate inputs

A fraction of `9/10` is mathematically exact. But it is conditional on ten claims being extracted and nine being correctly classified as supported.

An extractor might omit the most serious claim. A verifier might misread negation. A reference might be incomplete. The aggregation can be flawless while the evaluation result is misleading.

This is why a decimal score alone is not an explanation of evaluation reliability.

### 9.3 Averaging over a dataset introduces another decision

Suppose one response contains one claim and all of it is supported. Another contains 100 claims and 50 are supported.

The mean of response-level scores is:

```text
(1.0 + 0.5) / 2 = 0.75
```

The pooled claim-level support fraction is:

```text
(1 + 50) / (1 + 100) = 51/101 = approximately 0.505
```

The first gives each response equal weight. The second gives each claim equal weight. Neither is a mathematical mistake; they answer different aggregation questions.

Similarly, a system-level average can hide a small number of critical failures. A separate failure count communicates information that an average does not preserve.

### 9.4 Empty inputs and missing evidence are not universal zeros or ones

The expression `supported_claims / total_claims` is undefined when the denominator is zero. The metric implementation or application must decide how to represent that situation.

Different conditions also deserve different interpretations:

| Condition | Example interpretation, not a universal library standard |
|---|---|
| The output violates a verified requirement. | `FAIL`. |
| The requirement does not apply to this case. | `NOT_APPLICABLE`. |
| Required evidence is unavailable or genuinely inconclusive. | `INDETERMINATE`. |
| The evaluator times out or emits malformed results. | `ERROR`. |

Failure of the evaluator is not evidence that the candidate failed. Conversely, absence of extracted claims is not automatically evidence that a useful answer was produced.

### 9.5 Human agreement, repeatability, and correctness differ

**Human agreement** asks whether an evaluator matches reviewed judgments on a dataset. **Repeatability** asks whether it gives the same result across repeated executions. **Correctness** asks whether the result accurately reflects the task and evidence.

A repeatable evaluator can consistently apply the wrong rule. Two evaluators can agree because they share the same mistake. An evaluator can correlate well with overall human scores while missing a particular class of serious errors.

Meta-evaluation studies compare automated metrics with human judgments, including correlations and preference agreement. G-Eval, GPTScore, and the MT-Bench judge study use such analyses, but their findings remain tied to the evaluated tasks, data, and setups.[^geval][^gptscore][^mtbench]

### 9.6 Known conversation outcomes versus judge-validation labels

A stored conversation with a known outcome provides evidence about what happened. It does not automatically label the quality of every summary or intent result generated from it.

For the same conversation, Candidate A is faithful and sufficiently complete under the stated requirements, Candidate B changes the outcome, and Candidate C omits required state information. Those are output-level judgments derived from the source plus the output specification.

Thus two related datasets can exist:

```text
Known-outcome conversations:
    Source evidence for evaluating analyzer outputs.

Reviewed candidate outputs:
    Examples of correct and incorrect evaluator decisions
    used to inspect the evaluator itself.
```

Neither requires one uniquely worded ideal summary. Equivalent paraphrases can be acceptable while preserving the same required facts.

### 9.7 Testing what the evaluator is sensitive to

A conceptual way to inspect an evaluator is to vary one property at a time. Starting from Candidate A, changes could remove the word "no," replace the order identifier, omit the pending state, or restate the same facts with different wording.

The expected effects differ by metric. Removing a required fact should affect completeness. Introducing a false status should affect faithfulness. A faithful paraphrase should not be rejected merely because it differs from a reference's wording.

This tests the measurement's definition instead of assuming that any change in a score is desirable.

### 9.8 What makes an evaluation result interpretable?

A result is easier to interpret when its record identifies the method, model or checkpoint, criteria, evidence version, extraction/question policy, aggregation, and failure handling. For a source-grounded analyzer, preserving the assessed output and relevant source references makes the decision inspectable.

Model-provided explanations and evidence citations remain claims to check. A cited message can exist without supporting the conclusion.

The underlying principle is:

> A metric is a defined observation made through a procedure. Its name and numeric precision do not establish what it measures or how reliably it measures it.

---

## 10. Glossary

| Term | Meaning in this reference |
|---|---|
| Candidate | The generated output being evaluated. |
| Reference | An expected answer, summary, or other comparison target. |
| Source evidence | The material the candidate is supposed to represent or use. |
| Faithfulness | Support of the candidate's assertions by specified evidence. |
| Completeness | Preservation of the information required for the output's purpose. |
| Relevance | Whether included content addresses the specified task or focus. |
| Factual precision | A measure concerned with the factual claims the output makes. |
| Factual recall | A measure concerned with required/reference facts the output preserves. |
| Entailment | Whether an assertion follows from the supplied evidence. |
| Atomic claim | A small, self-contained factual assertion used as an evaluation unit. |
| Rubric | Descriptions mapping findings or quality levels to ratings. |
| Aggregation | Combining lower-level findings or scores into another result. |
| Calibration | Establishing how a score relates to observed outcomes in a defined setting. |
| Meta-evaluation | Evaluation of an evaluator's own behavior and reliability. |
| Reference-free | Not requiring a reference answer; source evidence may still be needed. |
| Deterministic scoring | A fixed mapping from specified inputs or findings to a result. |

---
