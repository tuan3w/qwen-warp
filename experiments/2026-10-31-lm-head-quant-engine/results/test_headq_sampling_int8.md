## sampling: Engine(head=int8) (R=64 rescoring) vs the BF16 head, N=1000000 draws per cell
  T=1.0 k=20 p=0.95 pos code+0   : support  2 top-1 p 0.927  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00010  (bf16 draws vs bf16 p: 0.00010)
  T=1.0 k=20 p=0.95 pos prose+0  : support  4 top-1 p 0.520  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00092  (bf16 draws vs bf16 p: 0.00092)
  T=1.0 k=20 p=0.95 pos math+0   : support  5 top-1 p 0.696  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00091  (bf16 draws vs bf16 p: 0.00091)
  T=1.0 k=20 p=0.95 pos prose+60 : support  2 top-1 p 0.932  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00004  (bf16 draws vs bf16 p: 0.00004)
  T=1.0 k=20 p=0.95 pos math+120 : support  1 top-1 p 1.000  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00000  (bf16 draws vs bf16 p: 0.00000)
  T=1.0 k=64 p=1.0 pos code+0   : support 64 top-1 p 0.898  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00048  (bf16 draws vs bf16 p: 0.00048)
  T=1.0 k=64 p=1.0 pos prose+0  : support 64 top-1 p 0.507  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00096  (bf16 draws vs bf16 p: 0.00096)
  T=1.0 k=64 p=1.0 pos math+0   : support 64 top-1 p 0.675  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00119  (bf16 draws vs bf16 p: 0.00119)
  T=1.0 k=64 p=1.0 pos prose+60 : support 64 top-1 p 0.930  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00009  (bf16 draws vs bf16 p: 0.00009)
  T=1.0 k=64 p=1.0 pos math+120 : support 64 top-1 p 0.988  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00005  (bf16 draws vs bf16 p: 0.00005)
  T=0.7 k=20 p=0.95 pos code+0   : support  1 top-1 p 1.000  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00000  (bf16 draws vs bf16 p: 0.00000)
  T=0.7 k=20 p=0.95 pos prose+0  : support  3 top-1 p 0.628  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00075  (bf16 draws vs bf16 p: 0.00075)
  T=0.7 k=20 p=0.95 pos math+0   : support  3 top-1 p 0.874  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00040  (bf16 draws vs bf16 p: 0.00040)
  T=0.7 k=20 p=0.95 pos prose+60 : support  1 top-1 p 1.000  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00000  (bf16 draws vs bf16 p: 0.00000)
  T=0.7 k=20 p=0.95 pos math+120 : support  1 top-1 p 1.000  analytic TV(q-head, bf16-head) 0.00000  empirical TV(q draws, bf16 p) 0.00000  (bf16 draws vs bf16 p: 0.00000)
- PASS  sampled-distribution TV vs the BF16 head < 0.005 on all 15 cells: worst analytic 0.00000, worst empirical 0.00119
RESULT: OK
