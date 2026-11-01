## E0 prompts, decode tok/s (HTTP, 385 tokens; median over rounds [min-max]; n = rounds)

**greedy**

| server | rounds | code | prose | math | 8K summarisation |
|---|---|---|---|---|---|
| ours | 2 | 165.6 [165-166] | 107.2 [107-107] | 165.7 [166-166] | 131.2 [131-132] |
| mtp-fast | 2 | 173.9 [174-174] | 104.1 [104-104] | 177.4 [177-177] | 139.2 [139-139] |
| dflash2 | 2 | 253.6 [253-254] | 102.9 [103-103] | 225.3 [225-226] | 121.3 [121-121] |

**T=1, top_p .95, top_k 20**

| server | rounds | code | prose | math | 8K summarisation |
|---|---|---|---|---|---|
| ours | 2 | 160.2 [157-164] | 107.8 [104-112] | 168.6 [163-175] | 139.1 [131-148] |
| mtp-fast | 2 | 157.5 [157-158] | 103.9 [104-104] | 169.9 [170-170] | 127.7 [128-128] |
| dflash2 | 2 | 213.8 [214-214] | 105.1 [105-105] | 225.2 [225-225] | 133.3 [133-133] |

## TTFT (ms; median over reps and rounds)

| server | 20-token prompt | 8K prompt (8,031 tok) |
|---|---|---|
| ours | 39 [39-41] | 3919 [3913-3927] |
| mtp-fast | 53 [34-56] | 4217 [4209-4222] |
| dflash2 | 56 [35-59] | 4161 [4148-4173] |

## Tokens per engine step and ms per step on the E0 prompts (greedy; HyperQwen: one SSE chunk = one engine step; swift-mk: one chunk = `--stream-cycles 2` = 2 cycles, shown per cycle)

| server | code tok/step | prose | math | 8K | code ms/step | prose | math | 8K |
|---|---|---|---|---|---|---|---|---|
| ours | 3.50 | 2.24 | 3.50 | 2.87 | 21.1 | 20.9 | 21.1 | 21.9 |
| mtp-fast | 3.81 | 2.29 | 3.89 | 3.21 | 21.9 | 22.0 | 21.9 | 23.0 |
| dflash2 | 5.83 | 2.39 | 5.20 | 2.98 | 23.0 | 23.2 | 23.1 | 24.6 |

## E6 tasks (32 prompts; a code edits, b log extraction, c agent tool-call turns, d prose, e math; 256 new tokens), greedy: decode tok/s (median over tasks of the class, median over rounds [min-max])

| server | rounds | a | b | c | d | e | pooled a-c | pooled all |
|---|---|---|---|---|---|---|---|---|
| ours | 2 | 197.4 [197-198] | 190.8 [191-191] | 240.4 [240-241] | 104.4 [104-104] | 164.0 [164-164] | 197 [197-198] | 185 [185-185] |
| mtp-fast | 2 | 157.5 [157-157] | 184.5 [184-185] | 184.5 [184-185] | 114.5 [114-115] | 178.6 [179-179] | 182 [182-182] | 173 [173-173] |
| dflash2 | 2 | 224.9 [225-225] | 225.2 [225-225] | 229.2 [229-229] | 110.2 [110-110] | 225.2 [225-225] | 225 [225-225] | 219 [219-219] |

## E6 tasks (32 prompts; a code edits, b log extraction, c agent tool-call turns, d prose, e math; 256 new tokens), T1: decode tok/s (median over tasks of the class, median over rounds [min-max])

| server | rounds | a | b | c | d | e | pooled a-c | pooled all |
|---|---|---|---|---|---|---|---|---|
| ours | 2 | 208.6 [203-215] | 182.6 [170-195] | 216.8 [207-227] | 110.2 [109-112] | 163.4 [162-165] | 205 [204-207] | 179 [178-179] |
| mtp-fast | 2 | 157.8 [158-158] | 176.2 [176-176] | 176.1 [176-176] | 105.4 [105-105] | 176.6 [177-177] | 171 [171-171] | 168 [167-168] |
| dflash2 | 2 | 203.9 [204-204] | 214.8 [215-215] | 223.0 [223-223] | 109.8 [110-110] | 227.0 [227-227] | 213 [213-213] | 207 [207-207] |

## E8 English tasks (48 prompts; 256 new tokens), greedy: decode tok/s (median over tasks of the class, median over rounds [min-max])

| server | rounds | ag | cd | em | ex | mt | qa | sm | tr | pooled all |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | 2 | 204.3 [204-204] | 153.9 [154-154] | 127.2 [127-127] | 135.3 [135-135] | 172.2 [172-172] | 126.0 [126-126] | 127.6 [128-128] | 134.4 [134-134] | 136 [136-136] |
| mtp-fast | 2 | 178.4 [178-178] | 174.8 [175-175] | 136.6 [137-137] | 129.9 [130-130] | 182.9 [183-183] | 129.9 [130-130] | 138.3 [138-138] | 162.2 [162-162] | 146 [146-146] |
| dflash2 | 2 | 238.9 [239-239] | 235.0 [235-235] | 135.9 [136-136] | 152.8 [153-153] | 240.2 [240-240] | 141.1 [141-141] | 149.7 [150-150] | 171.9 [172-172] | 168 [168-168] |

## E8 English tasks (48 prompts; 256 new tokens), T1: decode tok/s (median over tasks of the class, median over rounds [min-max])

| server | rounds | ag | cd | em | ex | mt | qa | sm | tr | pooled all |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | 2 | 207.2 [203-211] | 153.9 [153-155] | 130.0 [127-133] | 126.6 [126-127] | 173.3 [170-176] | 126.2 [125-127] | 125.4 [125-126] | 138.3 [136-141] | 135 [135-136] |
| mtp-fast | 2 | 171.1 [171-171] | 175.3 [175-175] | 127.7 [128-128] | 131.2 [131-131] | 182.4 [182-182] | 122.0 [122-122] | 137.1 [137-137] | 146.1 [146-146] | 140 [140-140] |
| dflash2 | 2 | 214.6 [215-215] | 239.6 [240-240] | 133.0 [133-133] | 140.8 [141-141] | 248.7 [249-249] | 129.0 [129-129] | 151.0 [151-151] | 165.6 [166-166] | 162 [162-162] |

## GPU state during the E0 decode phase (nvidia-smi 1 Hz while the GPU was active; median over rounds)

| server | SM clock MHz | power W (median) | power W (max) | temp C max | power limit W |
|---|---|---|---|---|---|
| ours | 1552 | 324 | 350 | 76 | 350 |
| mtp-fast | 1695 | 326 | 337 | 75 | 350 |
| dflash2 | 1680 | 325 | 335 | 74 | 350 |

boot to /health (s): ours 28, mtp-fast 38, dflash2 109

## Greedy parity: first 128 generated tokens, 30 E8 prompts (the first 30 tasks) + the 32 E6 prompts, same ids sent to both servers

| A vs B | set | prompts | identical 128 | mean common-prefix length (of 128) | median | first divergence at <= 16 tokens |
|---|---|---|---|---|---|---|
| ours vs mtp-fast | e8 | 30 | 7 | 59.2 | 48 | 5 |
| ours vs mtp-fast | e6 | 32 | 14 | 85.8 | 110 | 3 |
| ours vs dflash2 | e8 | 30 | 7 | 62.8 | 56 | 4 |
| ours vs dflash2 | e6 | 32 | 14 | 87.8 | 114 | 3 |
| mtp-fast vs dflash2 | e8 | 30 | 18 | 104.4 | 128 | 1 |
| mtp-fast vs dflash2 | e6 | 32 | 24 | 117.3 | 128 | 0 |
| ours vs ours (round 1 vs 2) | e8 | 30 | 30 | 128.0 | 128 | 0 |
| ours vs ours (round 1 vs 2) | e6 | 32 | 32 | 128.0 | 128 | 0 |
