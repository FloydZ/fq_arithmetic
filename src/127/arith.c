#include <stdio.h>
#include "arith.h"
#include "matrix.h"

#ifdef USE_AVX512
uint32_t test_gf127v_scalar_table() {
    __m512i table[2];
    gf127 tmp[64];
    for (uint32_t i = 1; i < 127; i++) {
        for (uint32_t j = 1; j <127; j++) {
            gf127v_scalar_u512_compute_table(table, j);

            for (uint32_t k = 0; k < 64; k++) { tmp[k] = i; }
            const __m512i a = _mm512_loadu_si512((const __m512i *)tmp);
            const __m512i c1 = gf127v_scalar_table_u512(a, table[0], table[1]);
            _mm512_storeu_si512((__m512i *)tmp, c1);

            const gf127 c = gf127_mul(i, j);
            for (uint32_t k = 0; k < 64; k++) {
                if (c != tmp[k]) {
                    printf("test_gf127v_scalar_table error\n");
                    return 1;
                }
            }
        } 
    }

    return 0;
}
#endif
uint32_t test_gf127_matrix_transpose8xN() {
    const uint32_t n = 8;
    const uint32_t m = 9;
    uint8_t *A  = (uint8_t *)malloc(n * m);
    uint8_t *B1 = (uint8_t *)malloc(n * m);

    // NOTE: this is important: LESS assumes that the input matrix is square
    uint8_t *B2 = (uint8_t *)malloc(m * m);
    for (uint32_t i = 0; i < n*m; i++) { A[i] = i; }

    gf127_matrix_transpose(B1, A, n, m);
    gf127_matrix_transpose8xN(B2, A, m);

    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            const uint8_t t1 = gf127_matrix_get(B1, n, i, j);
            const uint8_t t2 = gf127_matrix_get(B2, m, i, j);
            if (t1 != t2) {
                gf127_matrix_print(A, n, m);
                gf127_matrix_print(B1, m, n);
                gf127_matrix_print(B2, m, m);
                printf("error: test_gf127_matrix_transpose8xN: %d %d\n", i, j);
                return 1;
            }
        }
    }

    free(A); free(B1); free(B2);
    return 0;
}

uint32_t test_transpose(){
    const uint32_t n = 128;
    const uint32_t m = 128;
    uint8_t *A  = (uint8_t *)malloc(n * m);
    uint8_t *B1 = (uint8_t *)malloc(n * m);
    uint8_t *B2 = (uint8_t *)malloc(n * m);
    for (uint32_t i = 0; i < n*m; i++) { A[i] = i; }

    gf127_matrix_transpose(B1, A, n, m);
    gf127_matrix_transpose_opt2(B2, A, n);

    // gf127_matrix_print(B1, n, m);
    // gf127_matrix_print(B2, n, m);

    for (uint32_t i = 0; i < n*m; i++) {
        if (B1[i] != B2[i]) {
            printf("error: test_transpose: %d\n", i/n);
            return 1;
        }
    }

    free(A); free(B1); free(B2);
    return 0;
}

int main() {
#ifdef USE_AVX512
    if (test_gf127v_scalar_table()) { return 1; }
#endif

    // if (test_transpose()) { return 1; }
    // if (test_gf127_matrix_transpose8xN()) { return 1; }

    printf("all worked\n");
    return 0;
}
