Benchmarks:
==========


## Transpose 
Ryzen7600X, prefetching enabled:
```bash
blocksize = 64
BM_gf127_matrix_transpose_opt/64         834 ns          833 ns       841966
BM_gf127_matrix_transpose_opt/128       3314 ns         3308 ns       202898
BM_gf127_matrix_transpose_opt/192       7518 ns         7503 ns        93382
BM_gf127_matrix_transpose_opt/256      13611 ns        13585 ns        51551
BM_gf127_matrix_transpose_opt/320      21026 ns        20984 ns        33235
BM_gf127_matrix_transpose_opt/384      30289 ns        30231 ns        23116
BM_gf127_matrix_transpose_opt/448      41067 ns        40986 ns        17022
BM_gf127_matrix_transpose_opt/512      60378 ns        60259 ns        11639

blocksize = 32
BM_gf127_matrix_transpose_opt/64         874 ns          873 ns       800510
BM_gf127_matrix_transpose_opt/128       3712 ns         3704 ns       188293
BM_gf127_matrix_transpose_opt/192       8177 ns         8158 ns        85897
BM_gf127_matrix_transpose_opt/256      14535 ns        14505 ns        48237
BM_gf127_matrix_transpose_opt/320      22402 ns        22359 ns        31050
BM_gf127_matrix_transpose_opt/384      31862 ns        31802 ns        21977
BM_gf127_matrix_transpose_opt/448      44480 ns        44391 ns        15545
BM_gf127_matrix_transpose_opt/512      57992 ns        57868 ns        12103

blocksize = 16
BM_gf127_matrix_transpose_opt/64         900 ns          896 ns       773101
BM_gf127_matrix_transpose_opt/128       3513 ns         3506 ns       198849
BM_gf127_matrix_transpose_opt/192       8126 ns         8110 ns        87271
BM_gf127_matrix_transpose_opt/256      14701 ns        14673 ns        47707
BM_gf127_matrix_transpose_opt/320      22765 ns        22721 ns        30797
BM_gf127_matrix_transpose_opt/384      33353 ns        33283 ns        21075
BM_gf127_matrix_transpose_opt/448      45528 ns        45439 ns        15404
BM_gf127_matrix_transpose_opt/512      60295 ns        60182 ns        11630

blocksize = 8
BM_gf127_matrix_transpose_opt/64        1012 ns         1010 ns       721875
BM_gf127_matrix_transpose_opt/128       3843 ns         3836 ns       182304
BM_gf127_matrix_transpose_opt/192       8948 ns         8932 ns        78348
BM_gf127_matrix_transpose_opt/256      16347 ns        16317 ns        42928
BM_gf127_matrix_transpose_opt/320      24826 ns        24778 ns        28238
BM_gf127_matrix_transpose_opt/384      36462 ns        36391 ns        19234
BM_gf127_matrix_transpose_opt/448      48796 ns        48703 ns        14332
BM_gf127_matrix_transpose_opt/512      67001 ns        66863 ns        10519
```
