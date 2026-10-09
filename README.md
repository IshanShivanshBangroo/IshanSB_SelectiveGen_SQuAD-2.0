# Selective generation and abstention

Ishan Shivansh Bangroo · IST 597-002

This repository accompanies my tutorial on when an LLM application should answer, ask for clarification, or decline an unsupported answer.

[Open the notebook in Colab](https://colab.research.google.com/github/IshanShivanshBangroo/IshanSB_SelectiveGen_SQuAD-2.0/blob/main/IshanSBselective_generation_demo.ipynb)

[View the notebook and saved outputs](IshanSBselective_generation_demo.ipynb)

## Classroom replay

1. Open the Colab link.
2. Keep `RUN_GENERATION = False`.
3. Select **Runtime > Run all**.

The default run recomputes the analysis from the saved model outputs on a CPU. It does not query a model or require a GPU. The results are already displayed in the notebook for reading without execution. The original cache is also available as [results_cache.jsonl](results_cache.jsonl).

The main table uses fixed 50% expected coverage and a token-F1 correctness threshold of 0.5. Changing either fixed replay value stops with an explanatory message. The risk-coverage curves show other release levels. `SEED` controls the bootstrap resampling; changing it does not change the saved model responses.

A fresh model run is a separate option. It needs a suitable GPU, model and dataset downloads, and substantially more time than a classroom replay. Changing the model, prompts, precision, or software can change its outputs.

## Data and methods

The supplied run contains 418 items, with ten sampled answers per item:

- 400 SQuAD 2.0 validation questions, balanced between 200 answerable and 200 unanswerable questions.
- Six questions about the workshop notice used in class.
- Twelve constructed ambiguous questions, each with two supplied readings.

The cache records Qwen2.5-1.5B-Instruct on a Tesla T4 using float16, seed 597, temperature 1.0, and a 24-token answer limit. The recorded start time is 2026-10-09 06:17:22 UTC. This repository reruns the analysis of those outputs. It does not independently establish the training history of the model or exclude benchmark contamination.

The main table uses a deliberately permissive binary scoring rule: an answer to an answerable question is counted as correct when its token F1 with a gold answer is at least 0.5. A substantive answer to an unanswerable question is counted as wrong. This binary rule is **not the official SQuAD aggregate F1 or exact-match metric**. The notebook also reports exact-match sensitivity results.

The five gates select among the same fixed candidate answers. Their pooled comparison uses 50% expected coverage, with uniform random selection within a cutoff tie. Fractional release weights therefore describe expected counts, not a particular realized set of 200 releases. The separate held-out analysis sets a threshold and tie probability on 200 calibration items, then evaluates the other 200 items. Held-out coverage need not be exactly 50%. The question IDs are disjoint, but 26 passages appear in both halves and account for 29 test questions. This is a split by question, not a test on entirely new passages.

## Results from the saved run

| Policy | Coverage | Wrong among released | Correct baseline answers withheld |
| --- | ---: | ---: | ---: |
| Forced-answer prompt | 99.8% | 55.6% | 0.0% |
| Token probability gate | 50.0% | 46.0% | 39.0% |
| Verbalized confidence gate | 50.0% | 38.9% | 31.0% |
| Sample agreement gate | 50.0% | 40.8% | 33.1% |
| Lexical cluster entropy gate | 50.0% | 40.7% | 33.0% |
| Context-sufficiency self-check gate | 50.0% | 26.8% | 17.3% |

The forced-answer prompt released 399 of 400 answers, with 222 counted as wrong and 177 counted as correct. The last column measures the proportion of those 177 correct candidate answers withheld by each fixed-candidate gate.

The context-sufficiency self-check had the lowest observed risk among the tested gates on this sample. It asks whether the passage contains an answer, so it does not independently verify the selected answer. The notebook includes uncertainty intervals, risk-coverage curves, held-out thresholds, a random-selection comparison, an answerability oracle, and a separate prompt that permits an unanswerable response. The latter prompt changes the generated answers and is not a fixed-candidate gate.

The intervals are exploratory question-bootstrap summaries for one cached run. They are not guarantees for deployment, independent replications, or uncertainty over retraining and generation seeds. Questions can share passages, and comparisons were not adjusted for multiple testing.

## Lexical grouping and ambiguity

The local entropy signal uses greedy lexical grouping with a numerical-token guard. It is a **lexical-cluster proxy**, not a reproduction of the semantic entailment grouping used in the Nature paper. Numerical separation prevents examples such as “Room 204” and “Room 310” from being merged. Lexical matching still has limitations, and changing sample order can change greedy groups.

The twelve ambiguous items are constructed teaching examples with supplied readings. Results on these examples demonstrate how clarification can change the task. They do not estimate general ambiguity-detection accuracy or the quality of automatically generated interpretations. Class tally fields remain empty until the audience activity is performed.

## Quantum-information example

The tutorial's matrix example shows how a positive semidefinite similarity matrix can retain partial overlap between answer vectors. Its entropy agrees with ordinary entropy for the matching hard-group construction.

The cached experiment tests classical answer-withholding gates. It does not test a benefit from a spectral answer/ask/decline policy, require quantum hardware, or establish a quantum speed advantage. Any optional embedding-kernel analysis is exploratory and separate from the main result table.

## Reproducibility

The original cache is unchanged. Its SHA-256 is:

```
149f421f930223a53d163afa3186968bc8482497dfb0498e3af20b6e94cfae6a
```

The notebook contains the cache, the scoring rules, the thresholds, and the plots. The separate analysis script supports:

```bash
python analysis.py --cache results_cache.jsonl --output results_verified.json
```

The source data come from [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/). Dataset and model materials remain subject to their original terms.

## References

- Rajpurkar, P., Jia, R., and Liang, P. (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. *Proceedings of ACL, Volume 2*, 784-789. https://aclanthology.org/P18-2124/
- Yang, A., et al. (2024). Qwen2.5 Technical Report. arXiv:2412.15115. https://arxiv.org/abs/2412.15115
- Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature, 630*, 625-630. https://doi.org/10.1038/s41586-024-07421-0
- Hou, B., Liu, Y., Qian, K., Andreas, J., Chang, S., and Zhang, Y. (2024). Decomposing Uncertainty for Large Language Models through Input Clarification Ensembling. *Proceedings of ICML, PMLR 235*, 19023-19042. https://proceedings.mlr.press/v235/hou24b.html
- Nikitin, A., Kossen, J., Gal, Y., and Marttinen, P. (2024). Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities. *Advances in Neural Information Processing Systems, 37*. https://proceedings.neurips.cc/paper_files/paper/2024/hash/10c456d2160517581a234dfde15a7505-Abstract-Conference.html
- Walha, N., Gruber, S. G., Decker, T., Yang, Y., Javanmardi, A., Hüllermeier, E., and Buettner, F. (2026). Fine-grained uncertainty decomposition in large language models: A spectral approach. *Proceedings of the AAAI Conference on Artificial Intelligence, 40*(31), 26090-26098. https://doi.org/10.1609/aaai.v40i31.39811
