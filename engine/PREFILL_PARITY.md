# K6 fast prefill parity (tests/test_prefill.py; slow = prefill-by-decode T=4 chunks; fast = FastPrefill, fast_seg = segment size 512 / 1000)

## bf16 KV: oracle prompts + 1485-token topics prompt (--segs 512)
### code (32 prompt tokens, kv_fp8=0)
- first token slow 71093 fast 71093; prefill time slow 0.17s, fast 0.541s (first call includes triton autotune/compile)
- fast: last-token hidden rel-L2 vs slow: max over 64 layers 0.0184, final-norm 0.0187; DeltaNet state max/mean 0.0220/0.0117, conv state max 0.0263; KV K max/mean 0.0125/0.0087, V max/mean 0.0145/0.0101 (attention layer idx of max V: 13, per-layer V: 0.008 0.007 0.008 0.009 0.010 0.010 0.008 0.010 0.011 0.011 0.011 0.013 0.013 0.015 0.011 0.008); MTP KV K/V 0.0117/0.0108; drafts fast [71093, 12305, 198, 727] slow [71093, 12305, 198, 727]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0199, slow 0.0198
  fast: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (26) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
- fast_seg: last-token hidden rel-L2 vs slow: max over 64 layers 0.0184, final-norm 0.0187; DeltaNet state max/mean 0.0220/0.0117, conv state max 0.0263; KV K max/mean 0.0125/0.0087, V max/mean 0.0145/0.0101 (attention layer idx of max V: 13, per-layer V: 0.008 0.007 0.008 0.009 0.010 0.010 0.008 0.010 0.011 0.011 0.011 0.013 0.013 0.015 0.011 0.008); MTP KV K/V 0.0117/0.0108; drafts fast [71093, 12305, 198, 727] slow [71093, 12305, 198, 727]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0199, slow 0.0198
  fast_seg: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (26) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
### prose (34 prompt tokens, kv_fp8=0)
- first token slow 760 fast 760; prefill time slow 0.19s, fast 0.165s (first call includes triton autotune/compile)
- fast: last-token hidden rel-L2 vs slow: max over 64 layers 0.0131, final-norm 0.0133; DeltaNet state max/mean 0.0172/0.0109, conv state max 0.0201; KV K max/mean 0.0120/0.0090, V max/mean 0.0141/0.0107 (attention layer idx of max V: 13, per-layer V: 0.009 0.010 0.010 0.010 0.010 0.010 0.009 0.011 0.011 0.012 0.011 0.013 0.013 0.014 0.011 0.008); MTP KV K/V 0.0119/0.0111; drafts fast [760, 20513, 314, 279] slow [760, 20513, 314, 279]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0139, slow 0.0122
  fast: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (14) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
- fast_seg: last-token hidden rel-L2 vs slow: max over 64 layers 0.0131, final-norm 0.0133; DeltaNet state max/mean 0.0172/0.0109, conv state max 0.0201; KV K max/mean 0.0120/0.0090, V max/mean 0.0141/0.0107 (attention layer idx of max V: 13, per-layer V: 0.009 0.010 0.010 0.010 0.010 0.010 0.009 0.011 0.011 0.012 0.011 0.013 0.013 0.014 0.011 0.008); MTP KV K/V 0.0119/0.0111; drafts fast [760, 20513, 314, 279] slow [760, 20513, 314, 279]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0139, slow 0.0122
  fast_seg: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (14) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
### math (33 prompt tokens, kv_fp8=0)
- first token slow 760 fast 760; prefill time slow 0.17s, fast 0.165s (first call includes triton autotune/compile)
- fast: last-token hidden rel-L2 vs slow: max over 64 layers 0.0171, final-norm 0.0165; DeltaNet state max/mean 0.0200/0.0116, conv state max 0.0229; KV K max/mean 0.0118/0.0090, V max/mean 0.0136/0.0105 (attention layer idx of max V: 11, per-layer V: 0.009 0.008 0.008 0.009 0.010 0.010 0.009 0.010 0.011 0.012 0.013 0.014 0.013 0.013 0.011 0.008); MTP KV K/V 0.0115/0.0109; drafts fast [760, 4952, 421, 279] slow [760, 4952, 421, 279]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0158, slow 0.0141
  fast: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (30) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
- fast_seg: last-token hidden rel-L2 vs slow: max over 64 layers 0.0171, final-norm 0.0165; DeltaNet state max/mean 0.0200/0.0116, conv state max 0.0229; KV K max/mean 0.0118/0.0090, V max/mean 0.0136/0.0105 (attention layer idx of max V: 11, per-layer V: 0.009 0.008 0.008 0.009 0.010 0.010 0.009 0.010 0.011 0.012 0.013 0.014 0.013 0.013 0.011 0.008); MTP KV K/V 0.0115/0.0109; drafts fast [760, 4952, 421, 279] slow [760, 4952, 421, 279]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0158, slow 0.0141
  fast_seg: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (30) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
### long (1485 prompt tokens, kv_fp8=0)
- first token slow 8160 fast 8160; prefill time slow 6.96s, fast 0.781s (first call includes triton autotune/compile)
- fast: last-token hidden rel-L2 vs slow: max over 64 layers 0.0180, final-norm 0.0164; DeltaNet state max/mean 0.0273/0.0118, conv state max 0.0235; KV K max/mean 0.0282/0.0130, V max/mean 0.0261/0.0145 (attention layer idx of max V: 13, per-layer V: 0.008 0.008 0.008 0.009 0.010 0.011 0.010 0.012 0.015 0.016 0.017 0.020 0.020 0.026 0.023 0.020); MTP KV K/V 0.0256/0.0262; drafts fast [8160, 369, 264, 11782] slow [8160, 369, 264, 11782]
  fast: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (22) EQUAL
- fast_seg: last-token hidden rel-L2 vs slow: max over 64 layers 0.0186, final-norm 0.0188; DeltaNet state max/mean 0.0252/0.0117, conv state max 0.0262; KV K max/mean 0.0213/0.0111, V max/mean 0.0198/0.0124 (attention layer idx of max V: 13, per-layer V: 0.008 0.008 0.008 0.009 0.010 0.011 0.010 0.011 0.013 0.014 0.013 0.015 0.016 0.020 0.018 0.014); MTP KV K/V 0.0201/0.0191; drafts fast [8160, 369, 264, 11782] slow [8160, 369, 264, 11782]
  fast_seg: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (22) EQUAL
RESULT: PASS

## fp8 KV: same prompts (--fp8 1 --segs 512)
### code (32 prompt tokens, kv_fp8=1)
- first token slow 71093 fast 71093; prefill time slow 0.17s, fast 0.541s (first call includes triton autotune/compile)
- fast: last-token hidden rel-L2 vs slow: max over 64 layers 0.0281, final-norm 0.0283; DeltaNet state max/mean 0.0293/0.0159, conv state max 0.0333; KV K max/mean 0.0609/0.0505, V max/mean 0.0785/0.0593 (attention layer idx of max V: 14, per-layer V: 0.037 0.045 0.045 0.047 0.052 0.052 0.051 0.061 0.064 0.064 0.069 0.072 0.071 0.069 0.078 0.070); MTP KV K/V 0.0155/0.0148; drafts fast [71093, 12305, 198, 727] slow [71093, 12305, 198, 727]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0218, slow 0.0289
  fast: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (26) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
- fast_seg: last-token hidden rel-L2 vs slow: max over 64 layers 0.0281, final-norm 0.0283; DeltaNet state max/mean 0.0293/0.0159, conv state max 0.0333; KV K max/mean 0.0609/0.0505, V max/mean 0.0785/0.0593 (attention layer idx of max V: 14, per-layer V: 0.037 0.045 0.045 0.047 0.052 0.052 0.051 0.061 0.064 0.064 0.069 0.072 0.071 0.069 0.078 0.070); MTP KV K/V 0.0155/0.0148; drafts fast [71093, 12305, 198, 727] slow [71093, 12305, 198, 727]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0218, slow 0.0289
  fast_seg: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (26) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
### prose (34 prompt tokens, kv_fp8=1)
- first token slow 760 fast 760; prefill time slow 0.19s, fast 0.165s (first call includes triton autotune/compile)
- fast: last-token hidden rel-L2 vs slow: max over 64 layers 0.0156, final-norm 0.0155; DeltaNet state max/mean 0.0210/0.0138, conv state max 0.0244; KV K max/mean 0.0591/0.0483, V max/mean 0.0715/0.0576 (attention layer idx of max V: 14, per-layer V: 0.039 0.040 0.044 0.054 0.054 0.054 0.056 0.061 0.064 0.063 0.063 0.067 0.069 0.063 0.072 0.060); MTP KV K/V 0.0141/0.0141; drafts fast [760, 20513, 314, 279] slow [760, 20513, 314, 279]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0166, slow 0.0190
  fast: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (14) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
- fast_seg: last-token hidden rel-L2 vs slow: max over 64 layers 0.0156, final-norm 0.0155; DeltaNet state max/mean 0.0210/0.0138, conv state max 0.0244; KV K max/mean 0.0591/0.0483, V max/mean 0.0715/0.0576 (attention layer idx of max V: 14, per-layer V: 0.039 0.040 0.044 0.054 0.054 0.054 0.056 0.061 0.064 0.063 0.063 0.067 0.069 0.063 0.072 0.060); MTP KV K/V 0.0141/0.0141; drafts fast [760, 20513, 314, 279] slow [760, 20513, 314, 279]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0166, slow 0.0190
  fast_seg: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (14) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
### math (33 prompt tokens, kv_fp8=1)
- first token slow 760 fast 760; prefill time slow 0.17s, fast 0.166s (first call includes triton autotune/compile)
- fast: last-token hidden rel-L2 vs slow: max over 64 layers 0.0218, final-norm 0.0214; DeltaNet state max/mean 0.0313/0.0158, conv state max 0.0268; KV K max/mean 0.0681/0.0515, V max/mean 0.0806/0.0610 (attention layer idx of max V: 14, per-layer V: 0.034 0.045 0.046 0.048 0.051 0.052 0.046 0.066 0.063 0.067 0.075 0.077 0.081 0.078 0.081 0.068); MTP KV K/V 0.0159/0.0160; drafts fast [760, 4952, 421, 279] slow [760, 4952, 421, 279]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0217, slow 0.0248
  fast: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (30) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
- fast_seg: last-token hidden rel-L2 vs slow: max over 64 layers 0.0218, final-norm 0.0214; DeltaNet state max/mean 0.0313/0.0158, conv state max 0.0268; KV K max/mean 0.0681/0.0515, V max/mean 0.0806/0.0610 (attention layer idx of max V: 14, per-layer V: 0.034 0.045 0.046 0.048 0.051 0.052 0.046 0.066 0.063 0.067 0.075 0.077 0.081 0.078 0.081 0.068); MTP KV K/V 0.0159/0.0160; drafts fast [760, 4952, 421, 279] slow [760, 4952, 421, 279]
  vs bf16 oracle dump (last prompt token, max over layers): fast 0.0217, slow 0.0248
  fast_seg: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (30) EQUAL
  vs oracle gen_ids (9): fast EQUAL, slow EQUAL
### long (1485 prompt tokens, kv_fp8=1)
- first token slow 8160 fast 8160; prefill time slow 7.00s, fast 0.783s (first call includes triton autotune/compile)
- fast: last-token hidden rel-L2 vs slow: max over 64 layers 0.0221, final-norm 0.0229; DeltaNet state max/mean 0.0436/0.0173, conv state max 0.0364; KV K max/mean 0.0645/0.0541, V max/mean 0.0804/0.0637 (attention layer idx of max V: 14, per-layer V: 0.039 0.047 0.048 0.049 0.054 0.059 0.056 0.068 0.070 0.072 0.074 0.076 0.076 0.076 0.080 0.073); MTP KV K/V 0.0405/0.0395; drafts fast [8160, 369, 264, 11782] slow [8160, 369, 264, 11782]
  fast: first token EQUAL, plain greedy 33 ids DIFFER first at 17, spec k=3 ids (20) DIFFER
  fast plain divergence at generated index 17: slow logit gap top1 - other path's token 0.0508 -> NEAR-TIE (accepted)
  fast spec divergence at generated index 17: slow logit gap top1 - other path's token 0.0508 -> NEAR-TIE (accepted)
- fast_seg: last-token hidden rel-L2 vs slow: max over 64 layers 0.0240, final-norm 0.0250; DeltaNet state max/mean 0.0442/0.0163, conv state max 0.0342; KV K max/mean 0.0628/0.0529, V max/mean 0.0790/0.0623 (attention layer idx of max V: 14, per-layer V: 0.040 0.046 0.047 0.049 0.054 0.058 0.055 0.064 0.067 0.070 0.072 0.074 0.075 0.075 0.079 0.072); MTP KV K/V 0.0379/0.0384; drafts fast [8160, 369, 264, 11782] slow [8160, 369, 264, 11782]
  fast_seg: first token EQUAL, plain greedy 33 ids DIFFER first at 17, spec k=3 ids (20) DIFFER
  fast_seg plain divergence at generated index 17: slow logit gap top1 - other path's token 0.0256 -> NEAR-TIE (accepted)
  fast_seg spec divergence at generated index 17: slow logit gap top1 - other path's token 0.0256 -> NEAR-TIE (accepted)
RESULT: PASS

## bf16 KV: 3985-token prompt (--prompts long --long 4000 --ctx 4352 --segs 1000)
### long (3985 prompt tokens, kv_fp8=0)
- first token slow 8160 fast 8160; prefill time slow 18.74s, fast 2.305s (first call includes triton autotune/compile)
- fast: last-token hidden rel-L2 vs slow: max over 64 layers 0.0195, final-norm 0.0201; DeltaNet state max/mean 0.0318/0.0132, conv state max 0.0277; KV K max/mean 0.0451/0.0195, V max/mean 0.0572/0.0260 (attention layer idx of max V: 12, per-layer V: 0.008 0.008 0.008 0.009 0.010 0.011 0.010 0.013 0.019 0.022 0.054 0.054 0.057 0.055 0.047 0.031); MTP KV K/V 0.0413/0.0469; drafts fast [8160, 369, 264, 47193] slow [8160, 369, 264, 47193]
  fast: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (28) EQUAL
- fast_seg: last-token hidden rel-L2 vs slow: max over 64 layers 0.0212, final-norm 0.0213; DeltaNet state max/mean 0.0259/0.0126, conv state max 0.0267; KV K max/mean 0.0298/0.0138, V max/mean 0.0433/0.0178 (attention layer idx of max V: 13, per-layer V: 0.008 0.008 0.008 0.009 0.010 0.011 0.010 0.012 0.014 0.016 0.021 0.025 0.029 0.043 0.040 0.021); MTP KV K/V 0.0275/0.0297; drafts fast [8160, 369, 264, 47193] slow [8160, 369, 264, 47193]
  fast_seg: first token EQUAL, plain greedy 33 ids EQUAL, spec k=3 ids (28) EQUAL
RESULT: PASS

## noise floor (--baseline 1, 3985 tokens bf16; and fp8 1485 tokens): the same slow path with T=1 steps vs the T=4 slow path
- bf16 4K: hidden 0.0207, DeltaNet state 0.0258, conv 0.0233, K 0.0328 / V 0.0397 (max over layers), MTP KV 0.031/0.036, greedy 33 + spec 28 ids EQUAL
- fp8 1.5K: hidden 0.0241, DeltaNet state 0.0250, K 0.0539 / V 0.0666, MTP KV 0.021/0.020, plain greedy DIFFERS at generated index 17 (slow logit gap 0.0256; the fast path diverges at the same index to the same token)

# K10 short-prompt paths (tests/test_prefill.py; `w4_nofold` = W4 mma GEMM, last token as the T=1 head step; `w4` = + last row folded into the batch; `w4_seg` = 25-token segments; `fast*` = K6 dequant + cuBLAS; reference = slow prefill-by-decode)
Same checks as above; per-prompt detail is in the test output, condensed here (`fast_seg` = 512-token segments, long-prompt `w4` runs the whole 1485-token prompt through 64-row-class passes).  Hidden = max over 64 layers of the last prompt token's rel-L2 vs the slow path (taps written by `FastPrefill.taps` for the folded row).
### bf16 KV
| prompt | path | hidden rel-L2 max | DeltaNet state max | conv max | KV K/V max | MTP KV K/V | first token | plain 33 ids | spec ids |
|---|---|---|---|---|---|---|---|---|---|
| code (32) | fast | 0.0184 | 0.0220 | 0.0263 | 0.0125/0.0145 | 0.0117/0.0108 | EQUAL | EQUAL | EQUAL (26) |
| code (32) | fast_seg | 0.0184 | 0.0220 | 0.0263 | 0.0125/0.0145 | 0.0117/0.0108 | EQUAL | EQUAL | EQUAL (26) |
| code (32) | w4_nofold | 0.0205 | 0.0217 | 0.0258 | 0.0129/0.0149 | 0.0124/0.0115 | EQUAL | EQUAL | EQUAL (26) |
| code (32) | w4 | 0.0218 | 0.0219 | 0.0260 | 0.0129/0.0150 | 0.0124/0.0115 | EQUAL | EQUAL | EQUAL (26) |
| code (32) | w4_seg | 0.0197 | 0.0218 | 0.0265 | 0.0130/0.0152 | 0.0124/0.0116 | EQUAL | EQUAL | EQUAL (26) |
| prose (34) | fast | 0.0131 | 0.0172 | 0.0201 | 0.0120/0.0141 | 0.0119/0.0111 | EQUAL | EQUAL | EQUAL (14) |
| prose (34) | fast_seg | 0.0131 | 0.0172 | 0.0201 | 0.0120/0.0141 | 0.0119/0.0111 | EQUAL | EQUAL | EQUAL (14) |
| prose (34) | w4_nofold | 0.0128 | 0.0169 | 0.0196 | 0.0115/0.0138 | 0.0114/0.0107 | EQUAL | EQUAL | EQUAL (14) |
| prose (34) | w4 | 0.0128 | 0.0168 | 0.0196 | 0.0115/0.0138 | 0.0114/0.0107 | EQUAL | EQUAL | EQUAL (14) |
| prose (34) | w4_seg | 0.0121 | 0.0155 | 0.0188 | 0.0111/0.0133 | 0.0111/0.0104 | EQUAL | EQUAL | EQUAL (14) |
| math (33) | fast | 0.0171 | 0.0200 | 0.0229 | 0.0118/0.0136 | 0.0115/0.0109 | EQUAL | EQUAL | EQUAL (30) |
| math (33) | fast_seg | 0.0171 | 0.0200 | 0.0229 | 0.0118/0.0136 | 0.0115/0.0109 | EQUAL | EQUAL | EQUAL (30) |
| math (33) | w4_nofold | 0.0166 | 0.0204 | 0.0207 | 0.0116/0.0133 | 0.0110/0.0103 | EQUAL | EQUAL | EQUAL (30) |
| math (33) | w4 | 0.0157 | 0.0203 | 0.0207 | 0.0116/0.0132 | 0.0110/0.0103 | EQUAL | EQUAL | EQUAL (30) |
| math (33) | w4_seg | 0.0148 | 0.0200 | 0.0201 | 0.0118/0.0133 | 0.0109/0.0104 | EQUAL | EQUAL | EQUAL (30) |
| long (1485) | fast | 0.0180 | 0.0273 | 0.0235 | 0.0282/0.0261 | 0.0256/0.0262 | EQUAL | EQUAL | EQUAL (22) |
| long (1485) | fast_seg | 0.0186 | 0.0252 | 0.0262 | 0.0213/0.0198 | 0.0201/0.0191 | EQUAL | EQUAL | EQUAL (22) |
| long (1485) | w4_nofold | 0.0204 | 0.0246 | 0.0243 | 0.0220/0.0226 | 0.0213/0.0224 | EQUAL | EQUAL | EQUAL (22) |
| long (1485) | w4 | 0.0179 | 0.0246 | 0.0242 | 0.0220/0.0226 | 0.0213/0.0224 | EQUAL | EQUAL | EQUAL (22) |
| long (1485) | w4_seg | 0.0181 | 0.0260 | 0.0264 | 0.0321/0.0431 | 0.0298/0.0299 | EQUAL | EQUAL | EQUAL (22) |

- default FastPrefill (min_fast 2, w4_max 240), 4-token prompt (W4 mma path): spec first token + drafts EQUAL, plain first token EQUAL
- default FastPrefill (min_fast 2, w4_max 240), 6-token prompt (W4 mma path): spec first token + drafts EQUAL, plain first token EQUAL
- default FastPrefill (min_fast 2, w4_max 240), 32-token prompt (W4 mma path): spec first token + drafts EQUAL, plain first token EQUAL
- default FastPrefill (min_fast 2, w4_max 240), 200-token prompt (W4 mma path): spec first token + drafts EQUAL, plain first token EQUAL
- RESULT: PASS

### fp8 KV
| prompt | path | hidden rel-L2 max | DeltaNet state max | conv max | KV K/V max | MTP KV K/V | first token | plain 33 ids | spec ids |
|---|---|---|---|---|---|---|---|---|---|
| code (32) | fast | 0.0281 | 0.0293 | 0.0333 | 0.0609/0.0785 | 0.0155/0.0148 | EQUAL | EQUAL | EQUAL (26) |
| code (32) | fast_seg | 0.0281 | 0.0293 | 0.0333 | 0.0609/0.0785 | 0.0155/0.0148 | EQUAL | EQUAL | EQUAL (26) |
| code (32) | w4_nofold | 0.0270 | 0.0286 | 0.0351 | 0.0590/0.0724 | 0.0149/0.0141 | EQUAL | EQUAL | EQUAL (26) |
| code (32) | w4 | 0.0302 | 0.0298 | 0.0353 | 0.0597/0.0720 | 0.0149/0.0141 | EQUAL | EQUAL | EQUAL (26) |
| code (32) | w4_seg | 0.0254 | 0.0272 | 0.0324 | 0.0580/0.0721 | 0.0147/0.0137 | EQUAL | EQUAL | EQUAL (26) |
| prose (34) | fast | 0.0156 | 0.0210 | 0.0244 | 0.0591/0.0715 | 0.0141/0.0141 | EQUAL | EQUAL | EQUAL (14) |
| prose (34) | fast_seg | 0.0156 | 0.0210 | 0.0244 | 0.0591/0.0715 | 0.0141/0.0141 | EQUAL | EQUAL | EQUAL (14) |
| prose (34) | w4_nofold | 0.0160 | 0.0221 | 0.0248 | 0.0589/0.0714 | 0.0138/0.0134 | EQUAL | EQUAL | EQUAL (14) |
| prose (34) | w4 | 0.0204 | 0.0223 | 0.0252 | 0.0583/0.0709 | 0.0138/0.0134 | EQUAL | EQUAL | EQUAL (14) |
| prose (34) | w4_seg | 0.0191 | 0.0211 | 0.0224 | 0.0569/0.0688 | 0.0135/0.0128 | EQUAL | EQUAL | EQUAL (14) |
| math (33) | fast | 0.0218 | 0.0313 | 0.0268 | 0.0681/0.0806 | 0.0159/0.0160 | EQUAL | EQUAL | EQUAL (30) |
| math (33) | fast_seg | 0.0218 | 0.0313 | 0.0268 | 0.0681/0.0806 | 0.0159/0.0160 | EQUAL | EQUAL | EQUAL (30) |
| math (33) | w4_nofold | 0.0213 | 0.0389 | 0.0271 | 0.0684/0.0856 | 0.0160/0.0158 | EQUAL | EQUAL | EQUAL (30) |
| math (33) | w4 | 0.0229 | 0.0389 | 0.0275 | 0.0686/0.0861 | 0.0160/0.0158 | EQUAL | EQUAL | EQUAL (30) |
| math (33) | w4_seg | 0.0213 | 0.0372 | 0.0244 | 0.0665/0.0816 | 0.0149/0.0144 | EQUAL | EQUAL | EQUAL (30) |
| long (1485) | fast | 0.0221 | 0.0436 | 0.0364 | 0.0645/0.0804 | 0.0405/0.0395 | EQUAL | near-tie @17 | DIFFER (20) |
| long (1485) | fast_seg | 0.0240 | 0.0442 | 0.0342 | 0.0628/0.0790 | 0.0379/0.0384 | EQUAL | near-tie @17 | DIFFER (20) |
| long (1485) | w4_nofold | 0.0251 | 0.0478 | 0.0373 | 0.0652/0.0805 | 0.0475/0.0477 | EQUAL | near-tie @17 | DIFFER (20) |
| long (1485) | w4 | 0.0263 | 0.0478 | 0.0375 | 0.0652/0.0805 | 0.0475/0.0477 | EQUAL | near-tie @17 | DIFFER (20) |
| long (1485) | w4_seg | 0.0239 | 0.0301 | 0.0310 | 0.0590/0.0733 | 0.0273/0.0273 | EQUAL | EQUAL | DIFFER (20) |

- default FastPrefill (min_fast 2, w4_max 240), 4-token prompt (W4 mma path): spec first token + drafts EQUAL, plain first token EQUAL
- default FastPrefill (min_fast 2, w4_max 240), 6-token prompt (W4 mma path): spec first token + drafts EQUAL, plain first token EQUAL
- default FastPrefill (min_fast 2, w4_max 240), 32-token prompt (W4 mma path): spec first token + drafts EQUAL, plain first token EQUAL
- default FastPrefill (min_fast 2, w4_max 240), 200-token prompt (W4 mma path): spec first token + drafts EQUAL, plain first token EQUAL
- RESULT: PASS
