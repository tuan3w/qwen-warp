MTP layer quant int4clip, draft head int4 nv=65536 head_mode 0; spec buffers 5 MiB
### 1 packer
- packer clip=False: oracle-dequant vs packer-dequant max|diff| 0.00e+00; int4 rel-L2 vs source 0.1136
- packer clip=True: oracle-dequant vs packer-dequant max|diff| 0.00e+00; int4 rel-L2 vs source 0.1046
### 2 draft head vs full head
- restricted int4 head argmax == full BF16 head argmax: 100/108 (full-head argmax outside the 64k draft vocab: 4)
### 3 drafts vs oracle dump (BF16 MTP), forced a=1 cycles
- code: tokens EQUAL vs oracle gen_ids; draft ids == oracle BF16-MTP drafts: draft1 9/9, drafts1-2 7/9, drafts1-3 6/9; MTP hidden rel-L2 vs oracle (mean over 9 events) d1 0.2164 d2 0.3212 d3 0.5395
    mismatches (event, mine, oracle): [(1, [198, 464, 6673], [198, 12237, 198]), (3, [73111, 1393, 337], [73111, 1393, 25]), (4, [1393, 337, 76938], [1393, 25, 514])]
- prose: tokens EQUAL vs oracle gen_ids; draft ids == oracle BF16-MTP drafts: draft1 7/9, drafts1-2 5/9, drafts1-3 5/9; MTP hidden rel-L2 vs oracle (mean over 9 events) d1 0.2069 d2 0.4335 d3 0.6611
    mismatches (event, mine, oracle): [(0, [20513, 314, 279], [324, 55965, 557]), (2, [3851, 1323, 1012], [3851, 7906, 557]), (3, [5068, 6451, 557], [5068, 14593, 557]), (4, [15979, 1467, 524], [14593, 314, 5068])]
- math: tokens EQUAL vs oracle gen_ids; draft ids == oracle BF16-MTP drafts: draft1 8/9, drafts1-2 8/9, drafts1-3 8/9; MTP hidden rel-L2 vs oracle (mean over 9 events) d1 0.1980 d2 0.3713 d3 0.4445
    mismatches (event, mine, oracle): [(1, [421, 279, 2542], [310, 381, 16134])]
- total vs BF16 oracle: draft1 ids 24/27, drafts1-3 ids 19/27, mean d1 hidden rel-L2 0.207 (int4 MTP layer: expected ~0.2; engine vs the SAME int4 weights in the oracle = 0.014-0.016, tests/check_mtp_fq.py)
### 6 host-side EOS stop
- eos code: token 25 first at 6; spec stops with 7 tokens, prefix EQUAL
- eos prose: token 303 first at 2; spec stops with 3 tokens, prefix EQUAL
- eos math: token 16 first at 10; spec stops with 11 tokens, prefix EQUAL
### 4 speculative greedy == plain greedy (128 tokens)
- greedy code k=1: EQUAL (128 tokens); mean tokens/cycle 1.97, natural accepted-drafts histogram m=0..3 [2, 70, 0, 0] (1.5s)
- greedy code k=2: EQUAL (128 tokens); mean tokens/cycle 2.88, natural accepted-drafts histogram m=0..3 [2, 2, 44, 0] (1.1s)
- greedy code k=3: EQUAL (128 tokens); mean tokens/cycle 3.65, natural accepted-drafts histogram m=0..3 [2, 2, 4, 32] (1.0s)
- greedy prose k=1: NEAR-TIE divergence (accepted) at generated index 104 (plain 1049, spec 1216); plain-engine (prefix-prefill) logits top1 1049=22.8925 top2 1216=22.8271, margin 0.0655; mean tokens/cycle 1.65, natural accepted-drafts histogram m=0..3 [28, 52, 0, 0] (2.4s)
- greedy prose k=2: EQUAL (128 tokens); mean tokens/cycle 2.00, natural accepted-drafts histogram m=0..3 [25, 22, 25, 0] (1.6s)
- greedy prose k=3: NEAR-TIE divergence (accepted) at generated index 104 (plain 1049, spec 1216); plain-engine (prefix-prefill) logits top1 1049=22.8925 top2 1216=22.8271, margin 0.0655; mean tokens/cycle 2.06, natural accepted-drafts histogram m=0..3 [26, 16, 14, 8] (2.2s)
- greedy math k=1: EQUAL (128 tokens); mean tokens/cycle 1.90, natural accepted-drafts histogram m=0..3 [7, 65, 0, 0] (1.5s)
- greedy math k=2: EQUAL (128 tokens); mean tokens/cycle 2.67, natural accepted-drafts histogram m=0..3 [4, 8, 36, 0] (1.1s)
- greedy math k=3: EQUAL (128 tokens); mean tokens/cycle 3.23, natural accepted-drafts histogram m=0..3 [2, 10, 5, 23] (1.0s)
### 5 forced partial acceptance (rollback)
force pattern [1]
- rollback[1] code k=3: EQUAL (128 tokens); mean tokens/cycle 1.00, natural accepted-drafts histogram m=0..3 [3, 9, 12, 104] (2.8s)
- rollback[1] prose k=3: NEAR-TIE divergence (accepted) at generated index 104 (plain 1049, spec 1216); plain-engine (prefix-prefill) logits top1 1049=22.8925 top2 1216=22.8271, margin 0.0655; mean tokens/cycle 1.00, natural accepted-drafts histogram m=0..3 [38, 38, 29, 23] (3.5s)
- rollback[1] math k=3: EQUAL (128 tokens); mean tokens/cycle 1.00, natural accepted-drafts histogram m=0..3 [10, 16, 22, 80] (2.8s)
force pattern [2, 1, 3]
- rollback[2, 1, 3] code k=3: EQUAL (128 tokens); mean tokens/cycle 1.97, natural accepted-drafts histogram m=0..3 [3, 3, 6, 60] (1.7s)
- rollback[2, 1, 3] prose k=3: NEAR-TIE divergence (accepted) at generated index 104 (plain 1049, spec 1216); plain-engine (prefix-prefill) logits top1 1049=22.8925 top2 1216=22.8271, margin 0.0655; mean tokens/cycle 1.56, natural accepted-drafts histogram m=0..3 [28, 25, 19, 16] (2.7s)
- rollback[2, 1, 3] math k=3: EQUAL (128 tokens); mean tokens/cycle 1.88, natural accepted-drafts histogram m=0..3 [4, 11, 11, 46] (1.7s)
force pattern [1, 4, 2, 3, 1]
- rollback[1, 4, 2, 3, 1] code k=3: EQUAL (128 tokens); mean tokens/cycle 2.11, natural accepted-drafts histogram m=0..3 [1, 4, 6, 53] (1.5s)
- rollback[1, 4, 2, 3, 1] prose k=3: NEAR-TIE divergence (accepted) at generated index 104 (plain 1049, spec 1216); plain-engine (prefix-prefill) logits top1 1049=22.8925 top2 1216=22.8271, margin 0.0655; mean tokens/cycle 1.61, natural accepted-drafts histogram m=0..3 [25, 20, 22, 13] (2.5s)
- rollback[1, 4, 2, 3, 1] math k=3: EQUAL (128 tokens); mean tokens/cycle 2.09, natural accepted-drafts histogram m=0..3 [4, 9, 12, 39] (1.5s)
RESULT: packer=OK head=OK drafts=OK eos=OK greedy=OK rollback=OK
