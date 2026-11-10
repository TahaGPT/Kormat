# CPU probe (Intel i5-10310U, 4C/8T, fp32, torch CPU)

Each row is a fresh process; random token ids; greedy decode with KV cache; one repetition (indicative, other agents may load the CPU).

| config | component | threads | load_s | rss_after_load_mb | peak_rss_mb | prefill_256_ms | prefill_512_ms | prefill_1024_ms | decode_ms_per_token |
|---|---|---|---|---|---|---|---|---|---|
| target_base | target | 4 | 3.8 | 6229.0 | 9173.0 | 4123.0 | 8371.0 | 16594.0 | 515.0 |
| target_merged | target | 4 | 10.3 | 6337.0 | 9173.0 | 5004.0 | 9287.0 | 18922.0 | 496.3 |
| draft | draft | 4 | 3.4 | 2225.0 | 3166.0 | 1308.0 | 2840.0 | 5579.0 | 164.5 |
| target+draft | target | 4 | 9.5 | 8169.0 | 9173.0 | 4108.0 | 8483.0 | 17069.0 | 444.2 |
| target+draft | draft | 4 | 9.5 | 8169.0 | 9173.0 | 1237.0 | 2587.0 | 5262.0 | 150.8 |
| target_merged | target | 8 | 8.6 | 6336.0 | 9173.0 | 4651.0 | 9221.0 | 20550.0 | 488.9 |

## Recommendation
KORMAT_NUM_THREADS=4 (threads=4 unless another count is >10 % cheaper on prefill@512/24 + decode for the merged target; 4 vs 8 differed by 1.1 %, i.e. noise, so we stay on the 4 physical cores and leave headroom for other processes).

## Projected evaluation time
- Basis: criterion counts for val+test from results/split_manifest.json (412).
- Per criterion (merged target, 4 threads) = prefill@512 (9287.0 ms) + 24 decode tokens x 496.3 ms = 21.2 s (ASSUMPTIONS: 512-token prompt, 24-token answer; actual lengths come from the eval run).
- One model pass over 412 criteria: 146 min (2.4 h). Base+fine-tuned both on val+test: x2 = 4.9 h (sequential, no other load).
