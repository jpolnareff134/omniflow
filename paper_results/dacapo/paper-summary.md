| Benchmark | Policy | Reps | Sampling ratio | RMSE vs FUM (KB) |
|---|---|---:|---:|---:|
| cassandra | FUM | 10 | 100.0% ± 0.0% | 0.0 ± 0.0 |
| cassandra | ADP | 10 | 26.4% ± 1.3% | 1119.1 ± 283.4 |
| cassandra | INV | 10 | 28.0% ± 0.8% | 1136.2 ± 158.5 |
| cassandra | UNI | 10 | 50.0% ± 0.0% | 1257.8 ± 154.8 |
| cassandra | OmniFlow | 10 | 70.7% ± 0.4% | 1241.7 ± 153.7 |
| h2 | FUM | 10 | 100.0% ± 0.0% | 0.0 ± 0.0 |
| h2 | ADP | 10 | 43.6% ± 0.7% | 386.6 ± 63.5 |
| h2 | INV | 10 | 17.8% ± 0.9% | 412.4 ± 73.4 |
| h2 | UNI | 10 | 50.0% ± 0.0% | 342.5 ± 73.4 |
| h2 | OmniFlow | 10 | 22.1% ± 0.5% | 334.6 ± 64.0 |
| lusearch | FUM | 10 | 100.0% ± 0.0% | 0.0 ± 0.0 |
| lusearch | ADP | 10 | 49.8% ± 0.1% | 1689.9 ± 390.5 |
| lusearch | INV | 10 | 16.1% ± 0.2% | 3651.2 ± 557.7 |
| lusearch | UNI | 10 | 50.0% ± 0.0% | 3514.4 ± 858.6 |
| lusearch | OmniFlow | 10 | 100.0% ± 0.0% | 1380.3 ± 342.3 |
| tradebeans-release* | FUM | 10 | 100.0% ± 0.0% | 0.0 ± 0.0 |
| tradebeans-release* | ADP | 10 | 45.4% ± 0.7% | 397.7 ± 65.9 |
| tradebeans-release* | INV | 10 | 18.9% ± 1.0% | 389.4 ± 82.5 |
| tradebeans-release* | UNI | 10 | 50.0% ± 0.0% | 370.2 ± 62.2 |
| tradebeans-release* | OmniFlow | 10 | 25.9% ± 0.6% | 356.3 ± 81.5 |
| xalan-release* | FUM | 10 | 100.0% ± 0.0% | 0.0 ± 0.0 |
| xalan-release* | ADP | 10 | 4.4% ± 0.4% | 1310.7 ± 136.6 |
| xalan-release* | INV | 10 | 18.0% ± 0.2% | 1447.8 ± 175.9 |
| xalan-release* | UNI | 10 | 50.0% ± 0.0% | 1170.0 ± 157.2 |
| xalan-release* | OmniFlow | 10 | 22.6% ± 0.3% | 1077.2 ± 169.6 |

* `tradebeans-release` is the exact released companion-artifact arm. It is TPCC-backed and is not claimed to be the paper's described DayTrader workload.

* `xalan-release` uses the XML input filename as the stable request identity. The released source queues 17 inputs; the paper reports 16 Xalan request types.
