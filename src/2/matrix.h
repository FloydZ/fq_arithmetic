#pragma once

#include <string.h>
#include <stdint.h>
#include <stdio.h>

#include "arith.h"

#define MATRIX_AVX_PADDING(len) (((len + 255) / 256) * 256)
#define MAX_K           8
#define MAX_COLUMNS     256
#define WORD_SIZE       (8 * sizeof(uint64_t))


#define gf2_matrix_bytes_per_column(n_rows) (((n_rows) >> 3) + (((n_rows) & 0x1) | (((n_rows) & 0x2) >> 1) | (((n_rows) & 0x4) >> 2)))
#define gf2_matrix_bytes_size(n_rows, n_cols) ((gf2_matrix_bytes_per_column(n_rows)) * (n_cols))

/// precomputed with `table_precompute.c`
const uint8_t rev[9][256] __attribute__((aligned(64))) = {
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 3, 2, 7, 6, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10, 31, 30, 28, 29, 24, 25, 27, 26, 16, 17, 19, 18, 23, 22, 20, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10, 31, 30, 28, 29, 24, 25, 27, 26, 16, 17, 19, 18, 23, 22, 20, 21, 63, 62, 60, 61, 56, 57, 59, 58, 48, 49, 51, 50, 55, 54, 52, 53, 32, 33, 35, 34, 39, 38, 36, 37, 47, 46, 44, 45, 40, 41, 43, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10, 31, 30, 28, 29, 24, 25, 27, 26, 16, 17, 19, 18, 23, 22, 20, 21, 63, 62, 60, 61, 56, 57, 59, 58, 48, 49, 51, 50, 55, 54, 52, 53, 32, 33, 35, 34, 39, 38, 36, 37, 47, 46, 44, 45, 40, 41, 43, 42, 127, 126, 124, 125, 120, 121, 123, 122, 112, 113, 115, 114, 119, 118, 116, 117, 96, 97, 99, 98, 103, 102, 100, 101, 111, 110, 108, 109, 104, 105, 107, 106, 64, 65, 67, 66, 71, 70, 68, 69, 79, 78, 76, 77, 72, 73, 75, 74, 95, 94, 92, 93, 88, 89, 91, 90, 80, 81, 83, 82, 87, 86, 84, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10, 31, 30, 28, 29, 24, 25, 27, 26, 16, 17, 19, 18, 23, 22, 20, 21, 63, 62, 60, 61, 56, 57, 59, 58, 48, 49, 51, 50, 55, 54, 52, 53, 32, 33, 35, 34, 39, 38, 36, 37, 47, 46, 44, 45, 40, 41, 43, 42, 127, 126, 124, 125, 120, 121, 123, 122, 112, 113, 115, 114, 119, 118, 116, 117, 96, 97, 99, 98, 103, 102, 100, 101, 111, 110, 108, 109, 104, 105, 107, 106, 64, 65, 67, 66, 71, 70, 68, 69, 79, 78, 76, 77, 72, 73, 75, 74, 95, 94, 92, 93, 88, 89, 91, 90, 80, 81, 83, 82, 87, 86, 84, 85, 255, 254, 252, 253, 248, 249, 251, 250, 240, 241, 243, 242, 247, 246, 244, 245, 224, 225, 227, 226, 231, 230, 228, 229, 239, 238, 236, 237, 232, 233, 235, 234, 192, 193, 195, 194, 199, 198, 196, 197, 207, 206, 204, 205, 200, 201, 203, 202, 223, 222, 220, 221, 216, 217, 219, 218, 208, 209, 211, 210, 215, 214, 212, 213, 128, 129, 131, 130, 135, 134, 132, 133, 143, 142, 140, 141, 136, 137, 139, 138, 159, 158, 156, 157, 152, 153, 155, 154, 144, 145, 147, 146, 151, 150, 148, 149, 191, 190, 188, 189, 184, 185, 187, 186, 176, 177, 179, 178, 183, 182, 180, 181, 160, 161, 163, 162, 167, 166, 164, 165, 175, 174, 172, 173, 168, 169, 171, 170}
};

/// precompted 
const uint8_t diff[9][256] __attribute__((aligned(64))) = {
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 0, 2, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{ 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8}
};

/// \return a new matrix
static inline gf2* gf2_matrix_alloc(const uint32_t n_rows, 
                                    const uint32_t n_cols) {
    return (gf2 *)malloc(gf2_matrix_bytes_size(n_rows, n_cols));
}

/// \return a new matrix, with 256 cols
static inline gf2* gf2_matrix_alloc_padded(const uint32_t n_rows, 
                                           const uint32_t n_cols) {
    const uint32_t j = MATRIX_AVX_PADDING(n_cols);
    return (gf2 *)malloc(n_rows * j / 8);
}

///
/// \return matrix[i, j]
static inline gf2 gf2_matrix_get(const gf2 *matrix, 
                                 const uint32_t n_rows, 
                                 const uint32_t i,
                                 const uint32_t j) {
    const uint32_t nbytes_col = gf2_matrix_bytes_per_column(n_rows);
    const uint32_t idx_line = i / 8;
    const uint32_t bit_line = i % 8;

    return (matrix[nbytes_col * j + idx_line] >> bit_line) & 0x01;
}

/// NOTE: assumes row majow
/// \return matrix[i, j]
static inline gf2 gf2_matrix_get_row_major(const gf2 *matrix, 
                                           const uint32_t ncols, 
                                           const uint32_t i,
                                           const uint32_t j) {
    const uint32_t nbytes_row = gf2_matrix_bytes_per_column(ncols);
    const uint32_t idx_line = j / 8;
    const uint32_t bit_line = j % 8;

    return (matrix[ncols * i + idx_line] >> bit_line) & 0x01;
}


///  matrix[i, j] = scalar
static inline void gf2_matrix_set(gf2 *matrix,
                                  const uint32_t n_rows, 
                                  const uint32_t i,
                                  const uint32_t j,
                                  const gf2 scalar) {
    const uint32_t nbytes_col = gf2_matrix_bytes_per_column(n_rows);
    const uint32_t idx_line = i / 8;
    const uint32_t bit_line = i % 8;
    const uint8_t mask = 0xff ^ (1 << bit_line);
    matrix[nbytes_col*j + idx_line] = (matrix[nbytes_col*j + idx_line] & mask) ^ (scalar << bit_line);
}

/// @param matrix
/// @param n_rows
/// @param n_cols
static inline void gf2_matrix_rng(gf2 *matrix,
                                  const uint32_t n_rows,
                                  const uint32_t n_cols) {
    const uint32_t p = gf2_matrix_bytes_per_column(n_rows);
    for (uint32_t i = 0; i < n_cols; i++) {
        for (uint32_t j = 0; j < p; j++) {
            matrix[i*p + j] = rand();
        }
    }
}

/// NOTE: performs row operations
/// @param matrix
/// @param nrows
/// @param ncols
static inline void gf2_matrix_rng_full_rank(gf2 *matrix,
                                            const uint32_t nrows,
                                            const uint32_t ncols,
                                            const uint32_t padded) {
    const uint32_t p = gf2_matrix_bytes_per_column(nrows);
    for (uint32_t i = 0; i < ncols; i++) {
        for (uint32_t j = 0; j < nrows; j++) {
            gf2_matrix_set(matrix, nrows, i, j, i==j);
        }
    }

    for (uint32_t j1 = 0; j1 < nrows; j1++) {
        for (uint32_t j2 = 0; j2 < nrows; j2++) {
            if (j1 == j2) { continue; }

            const gf2 t = rand() % 2u;
            if (t) {
                // add j1 to j2
                for (uint32_t i = 0; i < ncols; i++) {
                    const gf2 s1 = gf2_matrix_get(matrix, nrows, j1, i);
                    const gf2 s2 = gf2_matrix_get(matrix, nrows, j2, i);
                    const gf2 s3 = s1^s3;
                    gf2_matrix_set(matrix, nrows, j2, i, s3);
                }
            }
        }
    }
}

///
/// @param matrix
/// @param n_rows
/// @param n_cols
static inline void gf2_matrix_zero(gf2 *matrix,
                                   const uint32_t n_rows,
                                   const uint32_t n_cols) {
    const uint32_t p = gf2_matrix_bytes_size(n_rows, n_cols);
    memset(matrix, 0, p);
}

///
/// @param matrix
/// @param n_rows
/// @param n_cols
static inline void gf2_matrix_set_to_ff(gf2 *matrix,
                                        const uint32_t n_rows, 
                                        const uint32_t n_cols) {
    if (n_rows & 0x7) {
        const uint32_t matrix_height =  gf2_matrix_bytes_per_column(n_rows);
        const uint32_t matrix_height_x = matrix_height -  1;

        const gf2 mask = 0xff >> (8 - (n_rows % 8));

        for (uint32_t i = 0; i < n_cols; i++) {
            matrix[i * matrix_height + matrix_height_x ] &= mask;
        }
    }
}

/// \param matrix: the matrix to print
/// \param nrows
/// \param ncols
static inline void gf2_matrix_print(const gf2 *matrix,
                                    const uint32_t nrows,
                                    const uint32_t ncols) {
    const uint32_t p = (ncols + 63) / 64;
    for (uint32_t i = 0; i < nrows; i++) {
        printf("[");
        for (uint32_t j = 0; j < p; j++) {
            for (uint32_t k = 0; k < 64; k++) {
                const uint32_t pos = j*64 + k;
                if (pos >= ncols) { break; }
                
                // const gf2 t = gf2_matrix_get_row_major(matrix, ncols, i, pos);
                const gf2 t = gf2_matrix_get(matrix, nrows, i, pos);
                if (t) {
                    printf("1");
                } else {
                    printf(" ");
                }

                if ((pos % 64 != 0) && (pos % 32 == 0)) {
                    printf(":");
                }
                if ((pos % 64 == 0) && (pos % 32 != 0)) {
                    printf("|");
                }
            }
        }
        printf("]\n");
    }
}

/// NOTE: row major
/// reads nn bits from row x and column y on
uint64_t matrix_read_bits(const uint8_t *M,
                          const uint32_t ncols,
                          const size_t x,
                          const size_t y,
                          const size_t nn) {
	const uint32_t spot  = y % WORD_SIZE;
	const uint64_t block = y / WORD_SIZE;
    const uint64_t *M2 = (uint64_t *)M;
    const uint32_t cp = (ncols + 63) / 64;

	// this must be negative...
	const int32_t spill = spot + nn - WORD_SIZE;
	uint64_t temp = (spill <= 0) ? M[x*cp + block + 0] << -spill
	                             :(M[x*cp + block + 1] << (WORD_SIZE - spill)) |
	                              (M[x*cp + block + 0] >> spill);
	return temp >> (WORD_SIZE - nn);
}

/// matrix1 = matrix2 + matrix3
static inline void gf2_matrix_add(gf2 *matrix1, 
                                  const gf2 *matrix2, 
                                  const gf2 *matrix3,
		                          const uint32_t n_rows, 
                                  const uint32_t n_cols) {
    const uint32_t n_bytes = gf2_matrix_bytes_size(n_rows, n_cols);

    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// matrix1 += scalar *matrix2
static inline void gf2_matrix_scalar_add(gf2 *matrix1, 
                                         const gf2 scalar, 
                                         const gf2 *matrix2,
                                         const uint32_t n_rows, 
                                         const uint32_t n_cols) {
    const uint32_t n_bytes = gf2_matrix_bytes_size(n_rows, n_cols);

    const uint8_t t = -scalar;
    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] ^= (t&matrix2[i]);
    }
}

/// result = matrix1 * matrix2
/// matrix1 of size n_rows1 * n_cols1
/// matrix2 of size n_cols1 * n_cols2
/// result  of size n_rows1 * n_cols2
static inline void gf2_matrix_mul(gf2 *result, 
                                  const gf2 *matrix1, 
                                  const gf2 *matrix2,
                                  const uint32_t n_rows1, 
                                  const uint32_t n_cols1, 
                                  const uint32_t n_cols2) {
    uint32_t i, j, k;
    gf2 entry_i_k, entry_k_j, entry_i_j;

    for (i = 0; i < n_rows1; i++) {
        for (j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (k = 0; k < n_cols1; k++) {
                entry_i_k = gf2_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf2_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2_mul(entry_i_k, entry_k_j);
            }

            gf2_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

#ifndef USE_AVX2
/// \param x
/// \param y
/// \param z = x ^ y
/// \param nn number of avx2 elements to add
inline void gf2_matrix_xor_rows(const uint8_t *x,
                                const uint8_t *y,
                                uint8_t *z,
                                const uint32_t nn) {
	for (uint32_t i = 0; i < nn; i += 1) {
		((uint64_t *)z)[4*i+0] = ((uint64_t *)x)[4*i+0] ^ ((uint64_t *)y)[4*i+0];
		((uint64_t *)z)[4*i+1] = ((uint64_t *)x)[4*i+1] ^ ((uint64_t *)y)[4*i+1];
		((uint64_t *)z)[4*i+2] = ((uint64_t *)x)[4*i+2] ^ ((uint64_t *)y)[4*i+2];
		((uint64_t *)z)[4*i+3] = ((uint64_t *)x)[4*i+3] ^ ((uint64_t *)y)[4*i+3];
    }
}
#endif

#ifdef USE_AVX2 

/// matrix1 = matrix2 + matrix3
static inline void gf2_matrix_add_u256(gf2 *matrix1, 
                                       const gf2 *matrix2, 
                                       const gf2 *matrix3,
		                               const uint32_t n_rows, 
                                       const uint32_t n_cols) {
    uint32_t n_bytes = gf2_matrix_bytes_size(n_rows, n_cols);
    while(n_bytes >= 32) {
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)matrix2);
        const __m256i t3 = _mm256_loadu_si256((const __m256i *)matrix3);
        const __m256i t4 = t2 ^ t3;
        _mm256_storeu_si256((__m256i *)matrix1, t4);
        n_bytes -= 32;
        matrix1 += 32;
        matrix2 += 32;
        matrix3 += 32;
    }

    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// matrix1 += scalar*matrix2
static inline void gf2_matrix_scalar_add_u256(gf2 *matrix1, 
                                              const gf2 scalar, 
                                              const gf2 *matrix2,
                                              const uint32_t n_rows, 
                                              const uint32_t n_cols) {
    uint32_t n_bytes = gf2_matrix_bytes_size(n_rows, n_cols);
    const uint8_t m = -scalar;
    const __m256i m256 = _mm256_set1_epi8(m);
    while(n_bytes >= 32) {
        const __m256i to = _mm256_loadu_si256((const __m256i *)matrix1);
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)matrix2);
        const __m256i t1 = to ^ (m256 & t2);
        _mm256_storeu_si256((__m256i *)matrix1, t1);
        n_bytes -= 32;
        matrix1 += 32;
        matrix2 += 32;
    }

    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] ^= m&matrix2[i];
    }
}

/// result = matrix1 * matrix2
/// matrix1 of size n_rows1 * n_cols1
/// matrix2 of size n_cols1 * n_cols2
/// result  of size n_rows1 * n_cols2
static inline void gf2_matrix_mul_u256(gf2 *result, 
                                       const gf2 *matrix1, 
                                       const gf2 *matrix2,
                                       const uint32_t n_rows1, 
                                       const uint32_t n_cols1, 
                                       const uint32_t n_cols2) {
    /// TODO
    uint32_t i, j, k;
    gf2 entry_i_k, entry_k_j, entry_i_j;

    for (i = 0; i < n_rows1; i++) {
        for (j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (k = 0; k < n_cols1; k++) {
                entry_i_k = gf2_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf2_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2_mul(entry_i_k, entry_k_j);
            }

            gf2_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// NOTE: assumes row major
/// \param x [in]
/// \param y [in]
/// \param z [out] z = x ^ y
/// \param nn number of avx2 registers
void gf2_matrix_xor_rows(const uint8_t *x,
                         const uint8_t *y,
                         uint8_t *z,
                         const uint32_t nn) {
	for (uint32_t i = 0; i < nn; i += 1) {
#ifdef USE_AVX2
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		__m256i x_avx = _mm256_load_si256((__m256i *)x + i);
		__m256i y_avx = _mm256_load_si256((__m256i *)y + i);
		__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
		_mm256_store_si256((__m256i *)z + i, z_avx);
#else
		__m256 x_avx = _mm256_loadu_ps((float*)x + 8*i);
		__m256 y_avx = _mm256_loadu_ps((float*)y + 8*i);
		__m256 z_avx = _mm256_xor_ps(x_avx,y_avx);
		_mm256_storeu_ps((float*)z + 8*i, z_avx);
#endif
#else
		((uint64_t *)z)[4*i] = ((uint64_t *)x)[4*i] ^ ((uint64_t *)y)[4*i];
		((uint64_t *)z)[4*i+1] = ((uint64_t *)x)[4*i+1] ^ ((uint64_t *)y)[4*i+1];
		((uint64_t *)z)[4*i+2] = ((uint64_t *)x)[4*i+2] ^ ((uint64_t *)y)[4*i+2];
		((uint64_t *)z)[4*i+3] = ((uint64_t *)x)[4*i+3] ^ ((uint64_t *)y)[4*i+3];
#endif
	}
}

/// NOTE: row major
/// swaps row i with j
/// \param M the matrix
/// \param ncols: number of columns 
/// \param i 
/// \param j 
void gf2_matrix_swap_rows_u256(uint8_t *__restrict__ M,
                               const uint32_t ncols,
                               const uint32_t i,
                               const uint32_t j) {
    if (i == j) { return; }
	const uint32_t cols_padded_ymm = MATRIX_AVX_PADDING(ncols) / 256;
    const uint32_t cp = (ncols+7) / 8;

	gf2_matrix_xor_rows((uint8_t *) M+(i*cp), (uint8_t *) M+(j*cp), (uint8_t *) M+(i*cp), cols_padded_ymm);
	gf2_matrix_xor_rows((uint8_t *) M+(j*cp), (uint8_t *) M+(i*cp), (uint8_t *) M+(j*cp), cols_padded_ymm);
	gf2_matrix_xor_rows((uint8_t *) M+(i*cp), (uint8_t *) M+(j*cp), (uint8_t *) M+(i*cp), cols_padded_ymm);
}

#endif

inline void gf2_matrix_swap_rows(uint8_t *__restrict__ M,
                                 const uint32_t ncols,
                                 const uint32_t i,
                                 const uint32_t j) {
    if (i == j) { return; }
	const uint32_t cols_padded_ymm = MATRIX_AVX_PADDING(ncols) / 256;
    const uint32_t cp = (ncols+7) / 8;

	gf2_matrix_xor_rows((uint8_t *) M+(i*cp), (uint8_t *) M+(j*cp), (uint8_t *) M+(i*cp), cols_padded_ymm);
	gf2_matrix_xor_rows((uint8_t *) M+(j*cp), (uint8_t *) M+(i*cp), (uint8_t *) M+(j*cp), cols_padded_ymm);
	gf2_matrix_xor_rows((uint8_t *) M+(i*cp), (uint8_t *) M+(j*cp), (uint8_t *) M+(i*cp), cols_padded_ymm);
}

/// NOTE assumes row major
/// \param M
/// \param r
/// \param c
/// \param nrows
/// \param ncols
/// \param k
static size_t matrix_gauss_submatrix(uint8_t *__restrict__ M,
                                     const size_t r,
                                     const size_t c,
                                     const size_t nrows,
                                     const size_t ncols,
                                     const size_t k) {
	size_t start_row = r, j;
	const uint32_t cols_padded_ymm = MATRIX_AVX_PADDING(ncols) / 256;
    const uint32_t cp = ncols / 8;

	for (j = c; j < c + k; ++j) {
		int found = 0;
		for (size_t i = start_row; i < nrows; ++i) {
			for (size_t l = 0; l < j - c; ++l) {
                if (gf2_matrix_get(M, ncols, i, c+l)) {
					gf2_matrix_xor_rows((uint8_t *) M+((r+l)*cp),
					                    (uint8_t *) M+(i*cp),
					                    (uint8_t *) M+(i*cp),
					                    cols_padded_ymm);
				}
			}

            if (gf2_matrix_get(M, ncols, i, j)) {
				gf2_matrix_swap_rows_u256(M, ncols, i, start_row);
				for (size_t l = r; l < start_row; ++l) {
                    if (gf2_matrix_get(M, ncols, l, j)) {
						gf2_matrix_xor_rows((uint8_t *) M+(start_row*cp),
						                    (uint8_t *) M+(l*cp),
						                    (uint8_t *) M+(l*cp),
						                    cols_padded_ymm);
					}
				}
				++start_row;
				found = 1;
				break;
			}
		}

		if (found == 0) {
			break;
		}
	}
	return j - c;
}

/// \param M the matrix in row major form
/// \param r
/// \param cols: number of cols
/// \param k m4ri 
/// \param T
/// \param diff
void matrix_make_table(uint8_t *__restrict__ M,
                       const size_t r,
                       const size_t cols,
                       const size_t k,
                       uint64_t *T,
                       uint32_t **diff) {
	const uint32_t cols_padded = MATRIX_AVX_PADDING(cols);
	const uint32_t cols_padded_word = cols_padded / 64;
	const uint32_t cols_padded_ymm = cols_padded / 256;
    const uint32_t cp = cols / 8;

	for (size_t i = 0; i < cols_padded_word; ++i) {
		T[i] = 0L;
	}

	for (size_t i = 0; i + 1 < 1UL << k; ++i) {
		gf2_matrix_xor_rows((uint8_t *)M+((r + diff[k][i])*cp),
		                    (uint8_t *)T,
		                    (uint8_t *)(T + cols_padded_word),
		                    cols_padded_ymm);
		T += cols_padded_word;
	}
}

/// \param
void matrix_process_rows(uint8_t *__restrict__ M,
                         const size_t rstart,
                         const size_t cstart,
                         const size_t rstop,
                         const size_t k,
                         const size_t cols,
                         uint64_t *T,
                         uint32_t **rev) {
	const uint32_t cols_padded = MATRIX_AVX_PADDING(cols);
	const uint32_t cols_padded_ymm = cols_padded / 256;
	const uint32_t cols_padded_word = cols_padded / 64;
    const uint32_t cp = (cols+7) / 8;

	for (size_t r = rstart; r < rstop; ++r) {
		size_t x0 = rev[k][matrix_read_bits(M, cols, r, cstart, k)];
		if (x0) {
			gf2_matrix_xor_rows((uint8_t *) (T + x0 * cols_padded_word),
			                    (uint8_t *) M+(r*cp),
			                    (uint8_t *) M+(r*cp),
			                    cols_padded_ymm);
		}
	}
}

///
/// \param M 			matrix
/// \param k 			m4ri k
/// \param rstop 		row stop
/// \param matrix_data 	helper data
/// \param cstart 		column start.
/// \return rank of the matrix
size_t gf2_matrix_echelonize(uint8_t *M,
                             const uint32_t nrows,
                             const uint32_t ncols,
                             const size_t k) {
    if (ncols > MAX_COLUMNS) { return -1; }

    /// NOTE: thats kind of crazy
    uint64_t xor_rows[MAX_COLUMNS/64 * (1<<MAX_K)];
	uint32_t kk = k;

	// current row
	uint32_t r = 0;

	// current column
	uint32_t c = 0;

	while (c < nrows) {
		if (c + kk > nrows) {
			kk = nrows - c;
		}

		size_t kbar = matrix_gauss_submatrix(M, r, c, nrows, ncols, kk);
		if (kk != kbar)
			break;

		if (kbar > 0) {
			matrix_make_table  (M, r, ncols, kbar, xor_rows, diff);
			// fix everything below
			matrix_process_rows(M, r + kbar, c, nrows, kbar, ncols, xor_rows, rev);
			// fix everything over it
			matrix_process_rows(M, 0, c, r, kbar, ncols, xor_rows, rev);
		}

		r += kbar;
		c += kbar;
	}

	return r;
}

/// original code from m4ri
/// Transpose a n x m matrix with width 1, offset 0 and m and n less than or equal 8.
/// Rows of all matrices are expected to have offset zero
/// and lay entirely inside a single block.
/// \note This function also works to transpose in-place.
/// \param dst First word of destination matrix.
/// \param src First word of source matrix.
/// \param rowstride_dst Rowstride of destination matrix.
/// \param rowstride_src Rowstride of source matrix.
/// \param n Number of rows in source matrix, must be less than or equal 8.
/// \param m Number of columns in source matrix, must be less than or equal 8.
/// \param maxsize TODO
void gf2_matrix_transpose_le8xle8(uint8_t *__restrict__ dst,
                                  uint8_t const *__restrict__ src,
                                  const uint32_t rowstride_dst, 
                                  const uint32_t rowstride_src, 
                                  int n, 
                                  int m,
                                  int maxsize) {
    int end                      = maxsize * 7;
    uint8_t const *restrict wks  = src;
    uint64_t w                   = *wks;
    int shift                    = 0;
    for (uint32_t i = 1; i < n; ++i) {
        wks += rowstride_src;
    	const uint8_t t1 = *wks;
    	const uint64_t t2 = t1;
        shift += 8;
        w |= (t2 << shift);
    }
    uint64_t mask = 0x80402010080402ULL;
    uint64_t w7   = w >> 7;
    shift         = 7;
    --m;

    do {
        uint64_t xor = (w ^ w7) & mask;
        mask >>= 8;
        w ^= (xor << shift);
        shift += 7;
        w7 >>= 7;
        w ^= xor;
    } while (shift < end);

    uint8_t *restrict wk = dst + m * rowstride_dst;
    for (int shift = 8 * m; shift > 0; shift -= 8) {
        *wk = (unsigned char)(w >> shift);
        wk -= rowstride_dst;
    }
    *wk = (unsigned char)w;
}
