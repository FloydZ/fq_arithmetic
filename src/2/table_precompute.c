#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "arith.h"

#define MATRIX_AVX_PADDING(len) (((len + 255) / 256) * 256)
#define WORD_SIZE (8 * sizeof(uint64_t))

/// \param
static int gray_new(const uint32_t i,
                    const uint32_t k) {
	int lastbit = 0;
	int res = 0;
	for (int j = k; j-- > 0;) {
		int bit = i & (1 << j);
		res |= (lastbit >> 1) ^ bit;
		lastbit = bit;
	}
	return res;
}

/// \param
void matrix_build_gray_code(uint32_t rev [MAX_K + 1][1u << MAX_K],
                            uint32_t diff[MAX_K + 1][1u << MAX_K]) {
	for (uint32_t k = 0; k <= MAX_K; ++k) {
		for (uint32_t i = 0; i < 1UL << k; ++i) {
			rev[k][gray_new(i, k)] = i;
		}

		for (uint32_t i = k + 1; i-- > 0;) {
			for (uint32_t j = 1; j < (1UL << i) + 1; ++j) {
				diff[k][j * (1 << (k - i)) - 1] = k - i;
			}
		}
	}
}

int main() {
	uint32_t rev [MAX_K + 1][1u << MAX_K];
	uint32_t diff[MAX_K + 1][1u << MAX_K];

    matrix_build_gray_code(rev, diff);

    // printf("const uint8_t rev[%d][%d] = {\n", MAX_K + 1, 1u << MAX_K);
	// for (uint32_t k = 0; k <= MAX_K; ++k) {
    //     printf("\t{ ");
    //     uint32_t i = 0;
	// 	for (; i < 1UL << k; ++i) {
    //         printf("%d", rev[k][i]);
    //         if (i < ((1ul << MAX_K) - 1)) {
    //             printf(", ");
    //         }
	// 	}
	// 	for (; i < 1ul << MAX_K; ++i) {
    //         printf("0");
    //         if (i < ((1ul << MAX_K) - 1)) {
    //             printf(", ");
    //         }
	// 	}
    //     printf("}");
    //     if (k < MAX_K) {
    //         printf(",\n");
    //     } else {
    //         printf("\n");
    //     }
	// }
    // printf("};\n");

    printf("const uint8_t diff[%d][%d] = {\n", MAX_K + 1, 1u << MAX_K);
	for (uint32_t k = 0; k <= MAX_K; ++k) {
        printf("\t{ ");
        uint32_t i = 0;
		for (; i < 1UL << k; ++i) {
            printf("%d", diff[k][i]);
            if (i < ((1ul << MAX_K) - 1)) {
                printf(", ");
            }
		}
		for (; i < 1ul << MAX_K; ++i) {
            printf("0");
            if (i < ((1ul << MAX_K) - 1)) {
                printf(", ");
            }
		}
        printf("}");
        if (k < MAX_K) {
            printf(",\n");
        } else {
            printf("\n");
        }
	}
    printf("};\n");
    return 0;
}
