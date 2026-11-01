### Main table (median of 5 interleaved repeats x 200 steps, 20 warmup; 15.086 GB streamed/step, 338 phases / 257 GEMV)

| variant | ms/step | rep medians (ms) | GB/s | % of V0 (read-only) | % of 936 GB/s | step p10-p90 (ms) |
|---|---|---|---|---|---|---|
| memcpy | 2.543 | 2.544, 2.543, 2.543, 2.544, 2.543 | 844.4 | 94.8 | 90.2 | 2.541-2.546 |
| v0 | 16.934 | 16.934, 16.934, 16.934, 16.933, 16.935 | 890.9 | 100.0 | 95.2 | 16.923-16.951 |
| v0g | 17.359 | 17.360, 17.359, 17.359, 17.359, 17.360 | 869.1 | 97.6 | 92.9 | 17.350-17.369 |
| v1a:w8r2d2 | 18.958 | 18.958, 18.966, 18.958, 18.957, 18.960 | 795.8 | 89.3 | 85.0 | 18.918-19.032 |
| v1b:w8r2d2 | 19.116 | 19.110, 19.126, 19.109, 19.130, 19.116 | 789.2 | 88.6 | 84.3 | 19.081-19.233 |
| v1c:w16r4d1 | 20.016 | 20.061, 20.016, 20.014, 20.014, 20.017 | 753.7 | 84.6 | 80.5 | 19.984-20.103 |
| v3:w8r2d2 | 18.922 | 18.920, 18.925, 18.922, 18.922, 18.938 | 797.3 | 89.5 | 85.2 | 18.886-19.001 |
| v4:w8r2d5 | 18.643 | 18.670, 18.669, 18.639, 18.643, 18.632 | 809.2 | 90.8 | 86.5 | 18.606-18.876 |
| v1a:w8r2d5 | 19.293 | 19.288, 19.293, 19.302, 19.309, 19.292 | 782.0 | 87.8 | 83.5 | 19.253-19.418 |
| v1b:w8r2d5 | 19.365 | 19.362, 19.357, 19.376, 19.380, 19.365 | 779.1 | 87.5 | 83.2 | 19.339-19.392 |
| v3:w8r2d5 | 19.168 | 19.156, 19.157, 19.171, 19.170, 19.168 | 787.1 | 88.3 | 84.1 | 19.142-19.195 |
| v4:w8r2d2 | 18.814 | 18.813, 18.814, 18.826, 18.821, 18.813 | 801.9 | 90.0 | 85.7 | 18.768-18.973 |

(memcpy row: bytes = read+write of a 1 GiB D2D copy.)

### Config sweep (2 repeats x 100 steps; info column = grid / smem)

| spec | ms/step | GB/s | info |
|---|---|---|---|
| v0 | 16.925 | 891.4 | single kernel, grid=246x512, 513 spans |
| v1a:w16r4d1 | 19.636 | 768.3 | grid=82 blocks/SM=1 smem=68608 |
| v1b:w16r4d1 | 19.546 | 771.9 | grid=82 blocks/SM=1 smem=68608 |
| v3:w16r4d1 | 19.371 | 778.8 | grid=82 blocks/SM=1 smem=68608 |
| v4:w16r4d1 | 19.253 | 783.6 | grid=82 blocks/SM=1 smem=68608 |
| v1a:w12r4d2 | 20.630 | 731.3 | grid=82 blocks/SM=1 smem=85504 |
| v1b:w12r4d2 | 20.542 | 734.4 | grid=82 blocks/SM=1 smem=85504 |
| v3:w12r4d2 | 20.376 | 740.4 | grid=82 blocks/SM=1 smem=85504 |
| v4:w12r4d2 | 19.907 | 757.9 | grid=82 blocks/SM=1 smem=85504 |
| v1a:w8r4d3 | 19.791 | 762.3 | grid=82 blocks/SM=1 smem=85504 |
| v1b:w8r4d3 | 19.520 | 772.9 | grid=82 blocks/SM=1 smem=85504 |
| v3:w8r4d3 | 19.365 | 779.0 | grid=82 blocks/SM=1 smem=85504 |
| v4:w8r4d3 | 18.798 | 802.6 | grid=82 blocks/SM=1 smem=85504 |
| v1a:w10r4d3 | 20.833 | 724.2 | grid=82 blocks/SM=1 smem=98176 |
| v1b:w10r4d3 | 20.327 | 742.2 | grid=82 blocks/SM=1 smem=98176 |
| v3:w10r4d3 | 20.173 | 747.9 | grid=82 blocks/SM=1 smem=98176 |
| v4:w10r4d3 | 19.569 | 770.9 | grid=82 blocks/SM=1 smem=98176 |
| v1a:w15r4d2 | 22.378 | 674.1 | grid=82 blocks/SM=1 smem=98176 |
| v1b:w15r4d2 | 22.260 | 677.7 | grid=82 blocks/SM=1 smem=98176 |
| v3:w15r4d2 | 22.004 | 685.6 | grid=82 blocks/SM=1 smem=98176 |
| v4:w15r4d2 | 21.486 | 702.1 | grid=82 blocks/SM=1 smem=98176 |
| v1a:w6r4d5 | 21.254 | 709.8 | grid=82 blocks/SM=1 smem=98176 |
| v1b:w6r4d5 | 20.858 | 723.3 | grid=82 blocks/SM=1 smem=98176 |
| v3:w6r4d5 | 20.546 | 734.3 | grid=82 blocks/SM=1 smem=98176 |
| v4:w6r4d5 | 19.784 | 762.6 | grid=82 blocks/SM=1 smem=98176 |
| v1a:w8r4d2 | 19.382 | 778.4 | grid=82 blocks/SM=1 smem=68608 |
| v1b:w8r4d2 | 19.392 | 778.0 | grid=82 blocks/SM=1 smem=68608 |
| v3:w8r4d2 | 19.237 | 784.2 | grid=82 blocks/SM=1 smem=68608 |
| v4:w8r4d2 | 18.985 | 794.6 | grid=82 blocks/SM=1 smem=68608 |
| v1a:w8r4d1 | 19.019 | 793.2 | grid=82 blocks/SM=1 smem=51712 |
| v1b:w8r4d1 | 19.182 | 786.5 | grid=82 blocks/SM=1 smem=51712 |
| v3:w8r4d1 | 18.983 | 794.7 | grid=82 blocks/SM=1 smem=51712 |
| v4:w8r4d1 | 18.992 | 794.4 | grid=82 blocks/SM=1 smem=51712 |
| v1a:w16r2d2 | 19.535 | 772.3 | grid=82 blocks/SM=1 smem=68608 |
| v1b:w16r2d2 | 19.430 | 776.4 | grid=82 blocks/SM=1 smem=68608 |
| v3:w16r2d2 | 19.298 | 781.8 | grid=82 blocks/SM=1 smem=68608 |
| v4:w16r2d2 | 19.095 | 790.1 | grid=82 blocks/SM=1 smem=68608 |
| v1a:w16r2d3 | 19.862 | 759.5 | grid=82 blocks/SM=1 smem=85504 |
| v1b:w16r2d3 | 19.658 | 767.5 | grid=82 blocks/SM=1 smem=85504 |
| v3:w16r2d3 | 19.512 | 773.2 | grid=82 blocks/SM=1 smem=85504 |
| v4:w16r2d3 | 19.044 | 792.2 | grid=82 blocks/SM=1 smem=85504 |
| v1a:w24r2d2 | 21.219 | 711.0 | grid=82 blocks/SM=1 smem=85504 |
| v1b:w24r2d2 | 20.993 | 718.6 | grid=82 blocks/SM=1 smem=85504 |
| v3:w24r2d2 | 21.062 | 716.3 | grid=82 blocks/SM=1 smem=85504 |
| v4:w24r2d2 | 21.147 | 713.4 | grid=82 blocks/SM=1 smem=85504 |
| v1a:w12r2d4 | 20.026 | 753.3 | grid=82 blocks/SM=1 smem=85504 |
| v1b:w12r2d4 | 19.846 | 760.2 | grid=82 blocks/SM=1 smem=85504 |
| v3:w12r2d4 | 19.635 | 768.4 | grid=82 blocks/SM=1 smem=85504 |
| v4:w12r2d4 | 19.111 | 789.4 | grid=82 blocks/SM=1 smem=85504 |
| v1a:w8r2d6 | 19.497 | 773.8 | grid=82 blocks/SM=1 smem=85504 |
| v1b:w8r2d6 | 19.581 | 770.5 | grid=82 blocks/SM=1 smem=85504 |
| v3:w8r2d6 | 19.290 | 782.1 | grid=82 blocks/SM=1 smem=85504 |
| v4:w8r2d6 | 18.724 | 805.7 | grid=82 blocks/SM=1 smem=85504 |
| v1a:w16r2d1 | 19.256 | 783.5 | grid=82 blocks/SM=1 smem=51712 |
| v1b:w16r2d1 | 19.207 | 785.5 | grid=82 blocks/SM=1 smem=51712 |
| v3:w16r2d1 | 19.166 | 787.2 | grid=82 blocks/SM=1 smem=51712 |
| v4:w16r2d1 | 19.082 | 790.6 | grid=82 blocks/SM=1 smem=51712 |
| v1a:w32r2d1 | 20.016 | 753.7 | grid=82 blocks/SM=1 smem=68608 |
| v1b:w32r2d1 | 19.852 | 759.9 | grid=82 blocks/SM=1 smem=68608 |
| v3:w32r2d1 | 22.460 | 671.7 | grid=82 blocks/SM=1 smem=68608 |
| v4:w32r2d1 | 23.846 | 632.7 | grid=82 blocks/SM=1 smem=68608 |
| v1a:w24r2d1 | 20.756 | 726.9 | grid=82 blocks/SM=1 smem=60160 |
| v1b:w24r2d1 | 20.624 | 731.5 | grid=82 blocks/SM=1 smem=60160 |
| v3:w24r2d1 | 20.509 | 735.6 | grid=82 blocks/SM=1 smem=60160 |
| v4:w24r2d1 | 20.602 | 732.3 | grid=82 blocks/SM=1 smem=60160 |
| v1a:w8r2d7 | 19.733 | 764.5 | grid=82 blocks/SM=1 smem=93952 |
| v1b:w8r2d7 | 19.605 | 769.5 | grid=82 blocks/SM=1 smem=93952 |
| v3:w8r2d7 | 19.392 | 778.0 | grid=82 blocks/SM=1 smem=93952 |
| v4:w8r2d7 | 18.720 | 805.9 | grid=82 blocks/SM=1 smem=93952 |
| v1a:w6r2d9 | 21.624 | 697.7 | grid=82 blocks/SM=1 smem=91840 |
| v1b:w6r2d9 | 21.815 | 691.6 | grid=82 blocks/SM=1 smem=91840 |
| v3:w6r2d9 | 21.890 | 689.2 | grid=82 blocks/SM=1 smem=91840 |
| v4:w6r2d9 | 21.323 | 707.5 | grid=82 blocks/SM=1 smem=91840 |
| v1a:w4r2d15 | 21.032 | 717.3 | grid=82 blocks/SM=1 smem=98176 |
| v1b:w4r2d15 | 21.652 | 696.8 | grid=82 blocks/SM=1 smem=98176 |
| v3:w4r2d15 | 20.820 | 724.6 | grid=82 blocks/SM=1 smem=98176 |
| v4:w4r2d15 | 20.480 | 736.7 | grid=82 blocks/SM=1 smem=98176 |
| v1a:w10r2d5 | 20.955 | 719.9 | grid=82 blocks/SM=1 smem=87616 |
| v1b:w10r2d5 | 21.030 | 717.4 | grid=82 blocks/SM=1 smem=87616 |
| v3:w10r2d5 | 20.580 | 733.1 | grid=82 blocks/SM=1 smem=87616 |
| v4:w10r2d5 | 20.101 | 750.5 | grid=82 blocks/SM=1 smem=87616 |
| v1a:w12r2d5 | 20.020 | 753.6 | grid=82 blocks/SM=1 smem=98176 |
| v1b:w12r2d5 | 19.941 | 756.5 | grid=82 blocks/SM=1 smem=98176 |
| v3:w12r2d5 | 19.605 | 769.5 | grid=82 blocks/SM=1 smem=98176 |
| v4:w12r2d5 | 18.956 | 795.8 | grid=82 blocks/SM=1 smem=98176 |
| v1a:w6r4d4 | 20.717 | 728.2 | grid=82 blocks/SM=1 smem=85504 |
| v1b:w6r4d4 | 20.488 | 736.3 | grid=82 blocks/SM=1 smem=85504 |
| v3:w6r4d4 | 20.364 | 740.8 | grid=82 blocks/SM=1 smem=85504 |
| v4:w6r4d4 | 19.798 | 762.0 | grid=82 blocks/SM=1 smem=85504 |
| v1a:w8r2d5 | 19.330 | 780.5 | grid=82 blocks/SM=1 smem=77056 |
| v1b:w8r2d5 | 19.357 | 779.4 | grid=82 blocks/SM=1 smem=77056 |
| v3:w8r2d5 | 19.167 | 787.1 | grid=82 blocks/SM=1 smem=77056 |
| v4:w8r2d5 | 18.673 | 807.9 | grid=82 blocks/SM=1 smem=77056 |
| v1a:w6r2d8 | 21.651 | 696.8 | grid=82 blocks/SM=1 smem=85504 |
| v1b:w6r2d8 | 21.823 | 691.3 | grid=82 blocks/SM=1 smem=85504 |
| v3:w6r2d8 | 21.985 | 686.2 | grid=82 blocks/SM=1 smem=85504 |
| v4:w6r2d8 | 21.452 | 703.3 | grid=82 blocks/SM=1 smem=85504 |
| v1a:w5r2d12 | 23.611 | 639.0 | grid=82 blocks/SM=1 smem=98176 |
| v1b:w5r2d12 | 24.523 | 615.2 | grid=82 blocks/SM=1 smem=98176 |
| v3:w5r2d12 | 24.043 | 627.5 | grid=82 blocks/SM=1 smem=98176 |
| v4:w5r2d12 | 23.418 | 644.2 | grid=82 blocks/SM=1 smem=98176 |
| v1a:w4r4d1 | 19.596 | 769.9 | grid=164 blocks/SM=2 smem=43264 |
| v1b:w4r4d1 | 19.407 | 777.4 | grid=164 blocks/SM=2 smem=43264 |
| v3:w4r4d1 | 19.407 | 777.4 | grid=164 blocks/SM=2 smem=43264 |
| v4:w4r4d1 | 19.552 | 771.6 | grid=164 blocks/SM=2 smem=43264 |
| v1a:w4r4d2 | 19.125 | 788.8 | grid=82 blocks/SM=1 smem=51712 |
| v1b:w4r4d2 | 19.231 | 784.5 | grid=82 blocks/SM=1 smem=51712 |
| v3:w4r4d2 | 19.073 | 791.0 | grid=82 blocks/SM=1 smem=51712 |
| v4:w4r4d2 | 18.950 | 796.1 | grid=82 blocks/SM=1 smem=51712 |
| v1a:w8r2d1 | 19.382 | 778.4 | grid=164 blocks/SM=2 smem=43264 |
| v1b:w8r2d1 | 19.275 | 782.7 | grid=164 blocks/SM=2 smem=43264 |
| v3:w8r2d1 | 19.139 | 788.3 | grid=82 blocks/SM=1 smem=43264 |
| v4:w8r2d1 | 19.206 | 785.5 | grid=82 blocks/SM=1 smem=43264 |
| v1a:w4r2d2 | 19.733 | 764.5 | grid=164 blocks/SM=2 smem=43264 |
| v1b:w4r2d2 | 19.687 | 766.3 | grid=164 blocks/SM=2 smem=43264 |
| v3:w4r2d2 | 19.624 | 768.8 | grid=164 blocks/SM=2 smem=43264 |
| v4:w4r2d2 | 19.620 | 768.9 | grid=164 blocks/SM=2 smem=43264 |
| v1a:w8r2d2 | 18.985 | 794.6 | grid=82 blocks/SM=1 smem=51712 |
| v1b:w8r2d2 | 19.148 | 787.9 | grid=82 blocks/SM=1 smem=51712 |
| v3:w8r2d2 | 18.951 | 796.1 | grid=82 blocks/SM=1 smem=51712 |
| v4:w8r2d2 | 18.818 | 801.7 | grid=82 blocks/SM=1 smem=51712 |
| v1a:w12r4d1 | 20.139 | 749.1 | grid=82 blocks/SM=1 smem=60160 |
| v1b:w12r4d1 | 20.243 | 745.3 | grid=82 blocks/SM=1 smem=60160 |
| v3:w12r4d1 | 19.948 | 756.3 | grid=82 blocks/SM=1 smem=60160 |
| v4:w12r4d1 | 19.918 | 757.4 | grid=82 blocks/SM=1 smem=60160 |
| v1a:w12r2d1 | 19.284 | 782.3 | grid=82 blocks/SM=1 smem=47488 |
| v1b:w12r2d1 | 19.290 | 782.1 | grid=82 blocks/SM=1 smem=47488 |
| v3:w12r2d1 | 19.156 | 787.5 | grid=82 blocks/SM=1 smem=47488 |
| v4:w12r2d1 | 19.157 | 787.5 | grid=82 blocks/SM=1 smem=47488 |
| v1a:w4r4d3 | 19.149 | 787.8 | grid=82 blocks/SM=1 smem=60160 |
| v1b:w4r4d3 | 19.271 | 782.9 | grid=82 blocks/SM=1 smem=60160 |
| v3:w4r4d3 | 19.097 | 790.0 | grid=82 blocks/SM=1 smem=60160 |
| v4:w4r4d3 | 18.890 | 798.6 | grid=82 blocks/SM=1 smem=60160 |
| v1a:w4r2d4 | 20.383 | 740.1 | grid=82 blocks/SM=1 smem=51712 |
| v1b:w4r2d4 | 20.847 | 723.7 | grid=82 blocks/SM=1 smem=51712 |
| v3:w4r2d4 | 20.285 | 743.7 | grid=82 blocks/SM=1 smem=51712 |
| v4:w4r2d4 | 20.561 | 733.7 | grid=82 blocks/SM=1 smem=51712 |
| v1c:w16r4d1 | 20.049 | 752.5 | grid=ceil(tasks/NW) CTAs per phase, min blocks/SM over phases=1 |
| v1c:w8r4d1 | 22.997 | 656.0 | grid=ceil(tasks/NW) CTAs per phase, min blocks/SM over phases=1 |
| v1c:w12r4d1 | 21.901 | 688.9 | grid=ceil(tasks/NW) CTAs per phase, min blocks/SM over phases=1 |
| v1c:w8r4d2 | 22.827 | 660.9 | grid=ceil(tasks/NW) CTAs per phase, min blocks/SM over phases=1 |
| v1c:w4r4d1 | 24.546 | 614.6 | grid=ceil(tasks/NW) CTAs per phase, min blocks/SM over phases=2 |
| v1c:w8r2d1 | 24.679 | 611.3 | grid=ceil(tasks/NW) CTAs per phase, min blocks/SM over phases=2 |

### Ablation of the persistent kernel (results invalid when abl!=0: bit0 no grid barrier, bit1 no activation prologue, bit2 no core phases)

| spec | ms/step | GB/s | info |
|---|---|---|---|
| v0 | 16.927 | 891.3 | single kernel, grid=246x512, 513 spans |
| v3:w8r2d2 | 18.916 | 797.6 | grid=82 blocks/SM=1 smem=51712 |
| v3:w8r2d2:abl=1 | 17.839 | 845.7 | grid=82 blocks/SM=1 smem=51712 |
| v3:w8r2d2:abl=2 | 18.407 | 819.6 | grid=82 blocks/SM=1 smem=51712 |
| v3:w8r2d2:abl=4 | 18.877 | 799.2 | grid=82 blocks/SM=1 smem=51712 |
| v3:w8r2d2:abl=7 | 17.767 | 849.1 | grid=82 blocks/SM=1 smem=51712 |
| v4:w8r2d5 | 18.629 | 809.8 | grid=82 blocks/SM=1 smem=77056 |
| v4:w8r2d5:abl=1 | 17.822 | 846.5 | grid=82 blocks/SM=1 smem=77056 |
| v4:w8r2d5:abl=2 | 18.100 | 833.5 | grid=82 blocks/SM=1 smem=77056 |
| v4:w8r2d5:abl=4 | 18.612 | 810.6 | grid=82 blocks/SM=1 smem=77056 |
| v4:w8r2d5:abl=7 | 17.786 | 848.2 | grid=82 blocks/SM=1 smem=77056 |

### Grid barrier microbenchmark (empty phases, 20000 barriers, median of 5)

| mode | blocks | threads | us/barrier |
|---|---|---|---|
| monotone_counter_spin | 82 | 128 | 0.798 |
| monotone_counter_nanosleep32 | 82 | 128 | 1.024 |
| sense_reversing | 82 | 128 | 1.127 |
| cg_grid_sync | 82 | 128 | 0.688 |
| monotone_counter_spin | 82 | 256 | 0.699 |
| monotone_counter_nanosleep32 | 82 | 256 | 1.024 |
| sense_reversing | 82 | 256 | 1.137 |
| cg_grid_sync | 82 | 256 | 0.693 |
| monotone_counter_spin | 82 | 512 | 0.736 |
| monotone_counter_nanosleep32 | 82 | 512 | 1.024 |
| sense_reversing | 82 | 512 | 1.174 |
| cg_grid_sync | 82 | 512 | 0.726 |
| monotone_counter_spin | 82 | 1024 | 0.843 |
| monotone_counter_nanosleep32 | 82 | 1024 | 1.024 |
| sense_reversing | 82 | 1024 | 1.286 |
| cg_grid_sync | 82 | 1024 | 0.831 |
| monotone_counter_spin | 164 | 128 | 0.856 |
| monotone_counter_nanosleep32 | 164 | 128 | 1.024 |
| sense_reversing | 164 | 128 | 1.201 |
| cg_grid_sync | 164 | 128 | 0.690 |
| monotone_counter_spin | 164 | 256 | 0.916 |
| monotone_counter_nanosleep32 | 164 | 256 | 1.024 |
| sense_reversing | 164 | 256 | 1.236 |
| cg_grid_sync | 164 | 256 | 0.715 |
| monotone_counter_spin | 164 | 512 | 1.042 |
| monotone_counter_nanosleep32 | 164 | 512 | 1.566 |
| sense_reversing | 164 | 512 | 1.374 |
| cg_grid_sync | 164 | 512 | 0.827 |

### Per-phase timeline of the persistent kernel (globaltimer, mean over phases of a kind and 20 steps; us)

**v3:w8r2d2** (grid 82)

| kind | phases/step | phase wall | start skew | prologue | stream (median block) | tail (last-median arrival) | barrier (last arrival->first release) | ms/step |
|---|---|---|---|---|---|---|---|---|
| core | 81 | 1.78 | 0.17 | 0.00 | 0.20 | 0.45 | 0.89 | 0.144 |
| down | 64 | 58.63 | 0.18 | 3.31 | 50.77 | 2.61 | 1.67 | 3.752 |
| gateup | 64 | 112.19 | 0.17 | 4.60 | 101.41 | 3.86 | 1.97 | 7.180 |
| lm_head | 1 | 2881.54 | 0.20 | 4.47 | 2842.14 | 34.41 | 0.00 | 2.882 |
| oproj | 64 | 22.32 | 0.18 | 2.28 | 17.81 | 0.89 | 0.94 | 1.429 |
| qkv_attn | 16 | 48.93 | 0.14 | 4.59 | 40.98 | 1.62 | 1.19 | 0.783 |
| qkvz_gdn | 48 | 56.23 | 0.18 | 4.59 | 46.83 | 3.29 | 1.25 | 2.699 |
| ALL | 338 | 55.82 | 0.18 | 2.81 | 49.24 | 2.15 | 1.31 | 18.869 |

**v4:w8r2d5** (grid 82)

| kind | phases/step | phase wall | start skew | prologue | stream (median block) | tail (last-median arrival) | barrier (last arrival->first release) | ms/step |
|---|---|---|---|---|---|---|---|---|
| core | 81 | 1.70 | 0.16 | 0.00 | 0.27 | 0.47 | 0.70 | 0.138 |
| down | 64 | 56.42 | 0.18 | 2.20 | 52.38 | 0.75 | 0.84 | 3.611 |
| gateup | 64 | 111.74 | 0.17 | 3.46 | 101.52 | 4.59 | 1.79 | 7.152 |
| lm_head | 1 | 2863.67 | 0.31 | 3.41 | 2835.69 | 23.96 | 0.00 | 2.864 |
| oproj | 64 | 21.77 | 0.19 | 1.10 | 18.52 | 0.76 | 0.77 | 1.393 |
| qkv_attn | 16 | 47.80 | 0.19 | 3.48 | 41.93 | 1.16 | 0.94 | 0.765 |
| qkvz_gdn | 48 | 55.06 | 0.20 | 3.47 | 47.18 | 2.87 | 1.17 | 2.643 |
| ALL | 338 | 54.93 | 0.18 | 1.95 | 49.79 | 1.80 | 1.02 | 18.565 |

**v4:w8r4d1** (grid 82)

| kind | phases/step | phase wall | start skew | prologue | stream (median block) | tail (last-median arrival) | barrier (last arrival->first release) | ms/step |
|---|---|---|---|---|---|---|---|---|
| core | 81 | 1.56 | 0.16 | 0.00 | 0.23 | 0.43 | 0.68 | 0.127 |
| down | 64 | 57.71 | 0.17 | 2.09 | 51.27 | 2.74 | 1.20 | 3.694 |
| gateup | 64 | 113.79 | 0.17 | 3.31 | 100.20 | 7.67 | 2.24 | 7.282 |
| lm_head | 1 | 2874.52 | 0.26 | 3.34 | 2828.88 | 41.32 | 0.00 | 2.875 |
| oproj | 64 | 21.94 | 0.18 | 1.04 | 18.43 | 1.35 | 0.87 | 1.404 |
| qkv_attn | 16 | 50.05 | 0.14 | 3.29 | 38.91 | 5.70 | 1.29 | 0.801 |
| qkvz_gdn | 48 | 57.70 | 0.18 | 3.31 | 45.89 | 6.73 | 1.35 | 2.770 |
| ALL | 338 | 56.07 | 0.17 | 1.86 | 48.95 | 3.68 | 1.23 | 18.952 |

**v3:w8r4d1** (grid 82)

| kind | phases/step | phase wall | start skew | prologue | stream (median block) | tail (last-median arrival) | barrier (last arrival->first release) | ms/step |
|---|---|---|---|---|---|---|---|---|
| core | 81 | 1.56 | 0.16 | 0.00 | 0.21 | 0.43 | 0.67 | 0.126 |
| down | 64 | 57.87 | 0.17 | 2.95 | 51.24 | 2.33 | 1.12 | 3.704 |
| gateup | 64 | 113.45 | 0.17 | 4.47 | 99.51 | 6.89 | 2.04 | 7.261 |
| lm_head | 1 | 2863.77 | 0.20 | 4.40 | 2813.95 | 44.75 | 0.00 | 2.864 |
| oproj | 64 | 22.27 | 0.17 | 2.11 | 17.60 | 1.20 | 0.87 | 1.425 |
| qkv_attn | 16 | 49.84 | 0.20 | 4.44 | 38.25 | 5.36 | 1.13 | 0.797 |
| qkvz_gdn | 48 | 57.45 | 0.16 | 4.45 | 45.13 | 6.07 | 1.21 | 2.758 |
| ALL | 338 | 56.02 | 0.17 | 2.66 | 48.47 | 3.33 | 1.15 | 18.935 |

### Single-shape GEMV bandwidth (graph of back-to-back launches over distinct matrices, incl. launch gaps); best 3 per shape + read-only

| shape | kernel | cfg | GB/s |
|---|---|---|---|
| gateup_34816x5120 | read_only_independent | - | 873.2 |
| gateup_34816x5120 | V1b | w4r4d1 | 846.3 |
| gateup_34816x5120 | V1a | w4r4d1 | 842.6 |
| gateup_34816x5120 | V1a | w16r4d1 | 842.1 |
| down_5120x17408 | read_only_independent | - | 859.8 |
| down_5120x17408 | V1b | w16r2d1 | 806.2 |
| down_5120x17408 | V1a | w16r2d1 | 803.1 |
| down_5120x17408 | V1a | w8r4d1 | 793.5 |
| qkvz_16384x5120 | read_only_independent | - | 859.7 |
| qkvz_16384x5120 | V1a | w4r4d1 | 808.5 |
| qkvz_16384x5120 | V1b | w4r4d1 | 805.6 |
| qkvz_16384x5120 | V1b | w8r2d1 | 804.6 |
| oproj_5120x6144 | read_only_independent | - | 816.2 |
| oproj_5120x6144 | V1b | w16r2d1 | 748.2 |
| oproj_5120x6144 | V1b | w8r4d1 | 747.6 |
| oproj_5120x6144 | V1b | w8r2d1 | 744.6 |
| lm_head_bf16_248320x5120 | read_only_independent | - | 891.2 |
| lm_head_bf16_248320x5120 | V1b | w15r4d2 | 888.5 |
| lm_head_bf16_248320x5120 | V1a | w15r4d2 | 888.4 |
| lm_head_bf16_248320x5120 | V1a | w12r2d5 | 887.7 |

### GEMV bandwidth vs CTA count (V1a, GB/s)

| shape | cfg | grid 41 | grid 64 | grid 82 |
|---|---|---|---|---|
| gateup_34816x5120 | w16r4d1 | 681.2 | 768.7 | 822.6 |
| gateup_34816x5120 | w8r4d1 | 655.6 | 847.3 | 825.3 |
| down_5120x17408 | w16r4d1 | 692.9 | 608.7 | 785.6 |
| down_5120x17408 | w8r4d1 | 648.8 | 732.6 | 790.7 |
| qkvz_16384x5120 | w16r4d1 | 651.7 | 811.3 | 727.1 |
| qkvz_16384x5120 | w8r4d1 | 647.4 | 820.2 | 785.0 |
| oproj_5120x6144 | w16r4d1 | 678.8 | 576.5 | 735.1 |
| oproj_5120x6144 | w8r4d1 | 609.0 | 683.8 | 743.0 |
| lm_head_bf16_248320x5120 | w16r4d1 | 883.4 | 881.1 | 884.0 |
| lm_head_bf16_248320x5120 | w8r4d1 | 866.6 | 883.2 | 882.7 |

### Correctness (full model vs naive fp32 reference)

{
 "reference": {
  "logits_rms": 0.9804,
  "x_rms": 1.8137,
  "reassoc_noise_floor_logits_rel_l2": 0.003773,
  "reassoc_noise_floor_x_rel_l2": 0.003087
 },
 "variants": [
  {
   "variant": "V1a",
   "cfg": "w8r2d5",
   "logits_rel_l2_vs_ref": 0.003791,
   "logits_max_abs_over_rms": 0.01847,
   "x_rel_l2_vs_ref": 0.003125,
   "finite": true,
   "logits_rel_l2_vs_first_variant": 0.0
  },
  {
   "variant": "V1b",
   "cfg": "w8r2d5",
   "logits_rel_l2_vs_ref": 0.003791,
   "logits_max_abs_over_rms": 0.01847,
   "x_rel_l2_vs_ref": 0.003125,
   "finite": true,
   "logits_rel_l2_vs_first_variant": 0.0
  },
  {
   "variant": "V3",
   "cfg": "w8r2d5",
   "logits_rel_l2_vs_ref": 0.003791,
   "logits_max_abs_over_rms": 0.01847,
   "x_rel_l2_vs_ref": 0.003125,
   "finite": true,
   "logits_rel_l2_vs_first_variant": 0.0
  },
  {
   "variant": "V4",
   "cfg": "w8r2d5",
   "logits_rel_l2_vs_ref": 0.003791,
   "logits_max_abs_over_rms": 0.01847,
   "x_rel_l2_vs_ref": 0.003125,
   "finite": true,
   "logits_rel_l2_vs_first_variant": 0.0
  }
 ]
}
