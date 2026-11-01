## e2e parity: Engine(head=int8) quantised head + rescoring vs the BF16 head in the same engine
- PASS  code plain greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
- PASS  code spec k=3 greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
  spec stats bf16 head: 3.661 tok/cycle hist [2, 4, 5, 45];  quantised: 3.661 tok/cycle hist [2, 4, 5, 45]
- PASS  prose plain greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
- PASS  prose spec k=3 greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
  spec stats bf16 head: 2.073 tok/cycle hist [37, 25, 24, 10];  quantised: 2.073 tok/cycle hist [37, 25, 24, 10]
- PASS  math plain greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
- PASS  math spec k=3 greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
  spec stats bf16 head: 3.344 tok/cycle hist [2, 15, 6, 41];  quantised: 3.344 tok/cycle hist [2, 15, 6, 41]
- PASS  e0_code plain greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
- PASS  e0_code spec k=3 greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
  spec stats bf16 head: 3.661 tok/cycle hist [2, 4, 5, 45];  quantised: 3.661 tok/cycle hist [2, 4, 5, 45]
- PASS  e0_prose plain greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
- PASS  e0_prose spec k=3 greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
  spec stats bf16 head: 2.073 tok/cycle hist [37, 25, 24, 10];  quantised: 2.073 tok/cycle hist [37, 25, 24, 10]
- PASS  e0_math plain greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
- PASS  e0_math spec k=3 greedy: 192 tokens, quantised-head stream IDENTICAL to the bf16-head stream
  spec stats bf16 head: 3.344 tok/cycle hist [2, 15, 6, 41];  quantised: 3.344 tok/cycle hist [2, 15, 6, 41]
RESULT: OK
