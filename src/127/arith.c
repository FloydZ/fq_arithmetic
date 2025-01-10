#include <stdio.h>
#include "arith.h"
#include "matrix.h"


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
            return 0;
        }
    }

    free(A); free(B1); free(B2);
    return 0;
}

int main() {
    if (test_transpose()) { return 1; }
    return 0;
}
