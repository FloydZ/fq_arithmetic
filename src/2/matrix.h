#pragma once

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include "arith.h"
#include "../helper.h"


#define MATRIX_AVX_PADDING(len) (((len + 255) / 256) * 256)
#define MAX_K           8
#define MAX_COLUMNS     256
#define WORD_SIZE       (8 * sizeof(uint64_t))


#define gf2_matrix_bytes_per_column(n_rows) (((n_rows) >> 3) + (((n_rows) & 0x1) | (((n_rows) & 0x2) >> 1) | (((n_rows) & 0x4) >> 2)))
#define gf2_matrix_bytes_size(n_rows, n_cols) ((gf2_matrix_bytes_per_column(n_rows)) * (n_cols))

///
/// \brief create a bit mask to zero out all but the (n - 1) % m4ri_radix + 1 leftmost bits.
///
/// This function returns 1..64 bits, never zero bits.
/// This mask is mainly used to mask the valid bits in the most significant word,
/// by using __M4RI_LEFT_BITMASK((M->ncols + M->offset) % m4ri_radix).
/// In other words, the set bits represent the columns with the lowest index in the word.
///
///  Thus,
///
///  n	Output
///  0=64 1111111111111111111111111111111111111111111111111111111111111111
///  1	  0000000000000000000000000000000000000000000000000000000000000001
///  2    0000000000000000000000000000000000000000000000000000000000000011
///  .                                     ...
/// 62    0011111111111111111111111111111111111111111111111111111111111111
/// 63	  0111111111111111111111111111111111111111111111111111111111111111
///
/// Note that n == 64 is only passed from __M4RI_MIDDLE_BITMASK, and still works
/// (behaves the same as n == 0): the input is modulo 64.
///
/// \param n Integer with 0 <= n <= m4ri_radix
///
#define __LEFT_BITMASK(n) ((-1ull) >> (64 - (n)) % 64)

/// NOTE: the max is only `MAX_M4RI_K` - 1
#define MAX_M4RI_K 9

/// precomputed with `table_precompute.c`
const uint8_t rev[MAX_M4RI_K][1u << (MAX_M4RI_K - 1)] __attribute__((aligned(64))) = {
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
const uint8_t diff[MAX_M4RI_K][1u << (MAX_M4RI_K - 1)] __attribute__((aligned(64))) = {
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

static unsigned char log2_ceil_table[64] = {
        0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};

static inline int log2_ceil(int n) { return log2_ceil_table[n - 1]; }

/// \return a new matrix
static inline gf2* gf2_matrix_alloc(const uint32_t n_rows, 
                                    const uint32_t n_cols) {
    return (gf2 *)calloc(1, gf2_matrix_bytes_size(n_rows, n_cols));
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
    const uint32_t idx_line = j / 8;
    const uint32_t bit_line = j % 8;

    return (matrix[(ncols + 7)/8 * i + idx_line] >> bit_line) & 0x01;
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
/// \param matrix
/// \param nrows
/// \param ncols
/// \param padded
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
                    const gf2 s3 = s1^s2;
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
/// \param matrix
/// \param n_rows
/// \param n_cols
static inline void gf2_matrix_id(gf2 *matrix,
                                 const uint32_t n_rows,
                                 const uint32_t n_cols) {
    gf2_matrix_zero(matrix, n_rows, n_cols);
    for (uint32_t  i = 0; i < MIN(n_rows, n_cols); ++i) {
        gf2_matrix_set(matrix, n_rows, i, i, 1);
    }
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

/// col major
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

/// row major
/// \param matrix: the matrix to print
/// \param nrows
/// \param ncols
static inline void gf2_matrix_print_row_major(const gf2 *matrix,
                                              const uint32_t nrows,
                                              const uint32_t ncols) {
    const uint32_t p = (ncols + 63) / 64;
    for (uint32_t i = 0; i < nrows; i++) {
        printf("[");
        for (uint32_t j = 0; j < p; j++) {
            for (uint32_t k = 0; k < 64; k++) {
                const uint32_t pos = j*64 + k;
                if (pos >= ncols) { break; }

                const gf2 t = gf2_matrix_get_row_major(matrix, ncols, i, pos);
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
	const uint32_t spot  = y % 8;
	const uint64_t block = y / 8;
    const uint32_t cp = (ncols + 7) / 8;
    const uint16_t mask = (1u << nn) - 1u;

    const uint16_t t1 = *(uint16_t *)(M + x*cp + block);
    const uint16_t t2 = t1 >> spot;
    const uint16_t t3 = t2 & mask;
	return t3;
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
#endif

/// NOTE: assumes row major
/// \param x [in]
/// \param y [in]
/// \param z [out] z = x ^ y
void gf2_matrix_xor_rows(const uint8_t *x,
                         const uint8_t *y,
                         uint8_t *z,
                         const uint32_t bytes) {
    uint32_t i = 0;
#ifdef USE_AVX2
	for (; i+32 <= bytes; i += 32) {
	    __m256 x_avx = _mm256_loadu_ps((float*)(x + i));
	    __m256 y_avx = _mm256_loadu_ps((float*)(y + i));
	    __m256 z_avx = _mm256_xor_ps(x_avx,y_avx);
	    _mm256_storeu_ps((float*)(z + i), z_avx);
	}
#endif
	for (; i+8 <= bytes; i += 8) {
		*((uint64_t *)(z+i)) = *((uint64_t *)(x+i)) ^ *((uint64_t *)(y+i));
	}

	for (; i < bytes; i++) {
	    z[i] = x[i] ^ y[i];
	}
}

/// NOTE: row major
/// swaps row i with j
/// \param M the matrix
/// \param ncols: number of columns
/// \param i
/// \param j
static void gf2_matrix_swap_rows(uint8_t *__restrict__ M,
                                 const uint32_t ncols,
                                 const uint32_t i,
                                 const uint32_t j) {
    if (i == j) { return; }
    const uint32_t cp = (ncols+7) / 8;

	gf2_matrix_xor_rows((uint8_t *) M+(i*cp), (uint8_t *) M+(j*cp), (uint8_t *) M+(i*cp), cp);
	gf2_matrix_xor_rows((uint8_t *) M+(j*cp), (uint8_t *) M+(i*cp), (uint8_t *) M+(j*cp), cp);
	gf2_matrix_xor_rows((uint8_t *) M+(i*cp), (uint8_t *) M+(j*cp), (uint8_t *) M+(i*cp), cp);
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
    const uint32_t cp = (ncols + 7) / 8;

	for (j = c; j < c + k; ++j) {
		int found = 0;
		for (size_t i = start_row; i < nrows; ++i) {
			for (size_t l = 0; l < j - c; ++l) {
                if (gf2_matrix_get_row_major(M, ncols, i, c+l)) {
					gf2_matrix_xor_rows((uint8_t *) M+((r+l)*cp),
					                    (uint8_t *) M+(i*cp),
					                    (uint8_t *) M+(i*cp),
					                    cp);
				}
			}

            if (gf2_matrix_get_row_major(M, ncols, i, j)) {
				gf2_matrix_swap_rows(M, ncols, i, start_row);
				for (size_t l = r; l < start_row; ++l) {
                    if (gf2_matrix_get_row_major(M, ncols, l, j)) {
						gf2_matrix_xor_rows((uint8_t *) M+(start_row*cp),
						                    (uint8_t *) M+(l*cp),
						                    (uint8_t *) M+(l*cp),
						                    cp);
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
                       const uint8_t diff[MAX_M4RI_K][1u << (MAX_M4RI_K - 1)]) {
	const uint32_t cols_padded_word = (cols + 63) / 64;
    const uint32_t cp = (cols + 7) / 8;

	for (size_t i = 0; i < cols_padded_word; ++i) {
		T[i] = 0L;
	}

	for (size_t i = 0; i + 1 < 1UL << k; ++i) {
		gf2_matrix_xor_rows((uint8_t *)M+((r + diff[k][i])*cp),
		                    (uint8_t *)T,
		                    (uint8_t *)(T + cols_padded_word),
		                    cp);
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
                         const uint64_t *T,
                         const uint8_t rev[MAX_M4RI_K][1u << (MAX_M4RI_K - 1)]) {
	const uint32_t cols_padded_word = (cols + 63) / 64;
    const uint32_t cp = (cols+7) / 8;

	for (size_t r = rstart; r < rstop; ++r) {
		const size_t x0 = rev[k][matrix_read_bits(M, cols, r, cstart, k)];
		if (x0) {
			gf2_matrix_xor_rows((uint8_t *) (T + x0 * cols_padded_word),
			                    (uint8_t *) M+(r*cp),
			                    (uint8_t *) M+(r*cp),
			                    cp);
		}
	}
}

///

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

static uint64_t const transpose_mask[6] = {
    0x5555555555555555ULL, 0x3333333333333333ULL, 0x0F0F0F0F0F0F0F0FULL,
    0x00FF00FF00FF00FFULL, 0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL,
};

/// code original from m4ri
/// Transpose 64/j matrices of size jxj in parallel.
/// 
/// Where j equals n rounded up to the nearest power of 2.
/// The input array t must be of size j (containing the rows i of all matrices in t[i]).
/// 
/// t[0..{j-1}]  = [Al]...[A1][A0]
/// 
/// \param t An array of j words.
/// \param n The number of rows in each matrix.
/// \return log2(j)
static inline int _mzd_transpose_Nxjx64(uint64_t *restrict t,
                                        int n) {
  int j  = 1;
  int mi = 0;  // Index into the transpose_mask array.

  while (j < n)  // Don't swap with entirely undefined data (where [D] exists entirely of
                 // non-existant rows).
  {
    // Swap 64/j matrices of size jxj in 2j rows. Thus,
    // <---- one word --->
    // [Al][Bl]...[A0][B0]
    // [Cl][Dl]...[C0][D0], where l = 64/j - 1 and each matrix [A], [B] etc is jxj.
    // Then swap [A] and [D] in-place.

    // m runs over the values in transpose_mask, so that at all
    // times m exists of j zeroes followed by j ones, repeated.
    uint64_t const m = transpose_mask[mi];
    int k        = 0;  // Index into t[].
    do {
      // Run over all rows of [A] and [D].
      for (int i = 0; i < j; ++i, ++k) {
        // t[k] contains row i of all [A], and t[k + j] contains row i of all [D]. Swap them.
        uint64_t xor = ((t[k] >> j) ^ t[k + j]) & m;
        t[k] ^= xor << j;
        t[k + j] ^= xor;
      }
      k += j;         // Skip [C].
    } while (k < n);  // Stop if we passed all valid input.

    // Double the size of j and repeat this for the next 2j rows until all
    // n rows have been swapped (possibly with non-existant rows).
    j <<= 1;
    ++mi;
  }

  return mi;
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
/// \param maxsize the max(n, m)
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
    for (uint32_t shift = 8 * m; shift > 0; shift -= 8) {
        *wk = (unsigned char)(w >> shift);
        wk -= rowstride_dst;
    }
    *wk = (unsigned char)w;
}

/// Rows of all matrices are expected to have offset zero
/// and lay entirely inside a single block.
/// 
/// \note This function also works to transpose in-place.
/// Transpose a n x m matrix with width 1, offset 0 and m and n less than or equal 16.
/// 
/// \param dst First word of destination matrix.
/// \param src First word of source matrix.
/// \param rowstride_dst Rowstride of destination matrix.
/// \param rowstride_src Rowstride of source matrix.
/// \param n Number of rows in source matrix, must be less than or equal 16.
/// \param m Number of columns in source matrix, must be less than or equal 16.
/// \param maxsize TODO
static inline void gf2_matrix_transpose_le16xle16(uint8_t *restrict dst,
                                                  uint8_t const *restrict src,
                                                  const uint32_t rowstride_dst, 
                                                  const uint32_t rowstride_src, 
                                                  int n,
                                                  int m, 
                                                  int maxsize) {
  int end                  = maxsize * 3;
  uint8_t const *restrict wks = (uint8_t *restrict)src;
  uint64_t t[4];
  int i = n;
  do {
    t[0] = *((uint16_t *)(wks+0));
    if (--i == 0) {
      t[1] = 0;
      t[2] = 0;
      t[3] = 0;
      break;
    }
    t[1] = *(uint16_t *)(wks+rowstride_src);
    if (--i == 0) {
      t[2] = 0;
      t[3] = 0;
      break;
    }
    t[2] = *(uint16_t *)(wks+2*rowstride_src);
    if (--i == 0) {
      t[3] = 0;
      break;
    }
    t[3] = *(uint16_t *)(wks+3*rowstride_src);
    if (--i == 0) break;
    wks += 4*rowstride_src;
    for (uint32_t shift = 16;; shift += 16) {
      t[0] |= ((uint64_t)(*(uint16_t *)wks) << shift);
      if (--i == 0) break;
      t[1] |= ((uint64_t)(*(uint16_t *)(wks + 1*rowstride_src)) << shift);
      if (--i == 0) break;
      t[2] |= ((uint64_t)(*(uint16_t *)(wks + 2*rowstride_src)) << shift);
      if (--i == 0) break;
      t[3] |= ((uint64_t)(*(uint16_t *)(wks + 3*rowstride_src)) << shift);
      if (--i == 0) break;
      wks += 4 * rowstride_src;
    }
  } while (0);
  uint64_t mask = 0xF0000F0000F0ULL;
  int shift = 12;
  uint64_t xor [4];
  do {
    xor[0] = (t[0] ^ (t[0] >> shift)) & mask;
    xor[1] = (t[1] ^ (t[1] >> shift)) & mask;
    xor[2] = (t[2] ^ (t[2] >> shift)) & mask;
    xor[3] = (t[3] ^ (t[3] >> shift)) & mask;
    mask >>= 16;
    t[0] ^= (xor[0] << shift);
    t[1] ^= (xor[1] << shift);
    t[2] ^= (xor[2] << shift);
    t[3] ^= (xor[3] << shift);
    shift += 12;
    t[0] ^= xor[0];
    t[1] ^= xor[1];
    t[2] ^= xor[2];
    t[3] ^= xor[3];
  } while (shift < end);
  _mzd_transpose_Nxjx64(t, 4);
  i                 = m;
  uint8_t *restrict wk = (uint8_t *)dst;
  do {
    *(uint16_t *)wk = (uint16_t)t[0];
    if (--i == 0) break;
    *(uint16_t *)(wk+rowstride_dst) = (uint16_t)t[1];
    if (--i == 0) break;
    *(uint16_t *)(wk+2*rowstride_dst) = (uint16_t)t[2];
    if (--i == 0) break;
    *(uint16_t *)(wk+3*rowstride_dst) = (uint16_t)t[3];
    if (--i == 0) break;
    wk += 4 * rowstride_dst;
    for (int shift = 16;; shift += 16) {
      wk[0] = (uint16_t)(t[0] >> shift);
      *(uint16_t *)wk = (uint16_t)(t[0] >> shift);
      if (--i == 0) break;
      wk[rowstride_dst] = (uint16_t)(t[1] >> shift);
      *(uint16_t *)(wk + rowstride_dst) = (uint16_t)(t[1] >> shift);
      if (--i == 0) break;
      *(uint16_t *)(wk + 2*rowstride_dst) = (uint16_t)(t[2] >> shift);
      if (--i == 0) break;
      *(uint16_t *)(wk + 3*rowstride_dst) = (uint16_t)(t[3] >> shift);
      if (--i == 0) break;
      wk += 4 * rowstride_dst;
    }
  } while (0);
}

/// Rows of all matrices are expected to have offset zero
/// and lay entirely inside a single block.
///
/// \note This function also works to transpose in-place.
/// Transpose a n x m matrix with width 1, offset 0 and m and n less than or equal 32.
///
/// \param dst First word of destination matrix.
/// \param src First word of source matrix.
/// \param rowstride_dst Rowstride of destination matrix.
/// \param rowstride_src Rowstride of source matrix.
/// \param n Number of rows in source matrix, must be less than or equal 32.
/// \param m Number of columns in source matrix, must be less than or equal 32.
static inline void gf2_matrix_transpose_le32xle32(uint8_t *restrict dst,
                                                  uint8_t const *restrict src,
                                                  const uint32_t rowstride_dst, 
                                                  const uint32_t rowstride_src, 
                                                  int n,
                                                  int m) {
  uint8_t const *restrict wks = (const uint8_t *)src;
  uint64_t t[16];

  const uint32_t mask = __LEFT_BITMASK(m);
  int i = n;
  if (n > 16) {
    i -= 16;
    for (uint32_t j = 0; j < 16; ++j) {
      t[j] = *(uint32_t *)wks;
      t[j] &= mask;
      wks += rowstride_src;
    }
    int j = 0;
    do {
      t[j++] |= (((uint64_t)*(uint32_t *)wks) << 32);
      wks += rowstride_src;
    } while (--i);
  } else {
    uint32_t j = 0;
    for (; j < n; ++j) {
      t[j] = *wks;
      wks += rowstride_src;
    }
    for (; j < 16; ++j) t[j] = 0;
  }
  _mzd_transpose_Nxjx64(t, 16);
  int one_more      = (m & 1);
  uint8_t *restrict wk = (uint8_t *)dst;
  if (m > 16) {
    m -= 16;
    for (int j = 0; j < 16; j += 2) {
      *(uint32_t *)wk				  = (t[j] & 0xFFFF) | ((t[j] >> 16) & 0xFFFF0000);
      *(uint32_t *)(wk+rowstride_dst) = (t[j + 1] & 0xFFFF) | ((t[j + 1] >> 16) & 0xFFFF0000);
      wk += 2 * rowstride_dst;
    }
    for (int j = 1; j < m; j += 2) {
      *(uint32_t *)wk				  = ((t[j - 1] >> 16) & 0xFFFF) | ((t[j - 1] >> 32) & 0xFFFF0000);
      *(uint32_t *)(wk+rowstride_dst) = ((t[j] >> 16) & 0xFFFF) | ((t[j] >> 32) & 0xFFFF0000);
      wk += 2 * rowstride_dst;
    }
    if (one_more) { *(uint32_t *)wk = ((t[m - 1] >> 16) & 0xFFFF) | ((t[m - 1] >> 32) & 0xFFFF0000); }
  } else {
    for (int j = 1; j < m; j += 2) {
      *(uint32_t *)wk                 = (t[j - 1] & 0xFFFF) | ((t[j - 1] >> 16) & 0xFFFF0000);
      *(uint32_t *)(wk+rowstride_dst) = (t[j] & 0xFFFF) | ((t[j] >> 16) & 0xFFFF0000);
      wk += 2 * rowstride_dst;
    }
    if (one_more) { *(uint32_t *)wk = (t[m - 1] & 0xFFFF) | ((t[m - 1] >> 16) & 0xFFFF0000); }
  }
}


/// Rows of both matrices are expected to fit exactly in a word (offset == 0)
/// and lay entirely inside a single block.
/// 
/// \note This function also works when dst == src.
/// 
/// Transpose a 64 x 64 matrix with width 1.
/// 
/// \param dst First word of destination matrix.
/// \param src First word of source matrix.
/// \param _rowstride_dst Rowstride of matrix dst.
/// \param _rowstride_src Rowstride of matrix src.
static inline void gf2_matrix_transpose_64x64(uint8_t *dst,
                                              uint8_t const *src,
                                              const uint32_t _rowstride_dst,
                                              const uint32_t _rowstride_src) {
    /// m runs over the values:
    ///   0x00000000FFFFFFFF
    ///   0x0000FFFF0000FFFF
    ///   0x00FF00FF00FF00FF
    ///   0x0F0F0F0F0F0F0F0F
    ///   0x3333333333333333
    ///   0x5555555555555555,
    /// alternating j zeroes with j ones.
    ///
    /// Assume we have a matrix existing of four jxj matrices ((0,0) is in the top-right corner,
    /// this is the memory-model view, see the layout on
    /// http://m4ri.sagemath.org/doxygen/structmzd__t.html):
    /// ...[A1][B1][A0][B0]
    /// ...[C1][D1][C0][D0]
    ///          . [A2][B2]
    ///        .   [C2][B2]
    ///      .         .
    ///                .
    /// The following calculates the XOR between A and D,
    /// and subsequently applies that to A and D respectively,
    /// swapping A and D as a result.
    /// Therefore, wk starts at the first row and then has rowstride
    /// added j times, running over the rows of A, then skips C

    const uint32_t rowstride_dst = (_rowstride_dst + 7)/8;
    const uint32_t rowstride_src = (_rowstride_src + 7)/8;
    uint64_t m               = (uint64_t)0xFFFFFFFF;
    uint32_t j_rowstride_dst = rowstride_dst * 64;
    uint32_t j_rowstride_src = rowstride_src * 32;
    uint64_t *const end      = ((uint64_t *)(dst)) + j_rowstride_dst;
    // We start with j = 32, and a one-time unrolled loop, where
    // we copy from src and write the result to dst, swapping
    // the two 32x32 corner matrices.
    int j = 32;
    j_rowstride_dst >>= 1;
    uint64_t *restrict wk = (uint64_t *)dst;
    for (uint64_t const *restrict wks = (uint64_t *)src; wk < end; wk += j_rowstride_dst, wks += j_rowstride_src) {
        for (int k = 0; k < j; ++k, wk += rowstride_dst, wks += rowstride_src) {
            uint64_t xor            = ((*wks >> j) ^ *(wks + j_rowstride_src)) & m;
            *wk                     = *wks ^ (xor << j);
            *(wk + j_rowstride_dst) = *(wks + j_rowstride_src) ^ xor;
        }
    }

    // Next we work in-place in dst and swap the corners of
    // each of the last matrices, all in parallel, for all
    // remaining values of j.
    m ^= m << 16;
    for (j = 16; j != 0; j = j >> 1, m ^= m << j) {
        j_rowstride_dst >>= 1;
        for (wk = (uint64_t *)dst; wk < end; wk += j_rowstride_dst) {
            for (int k = 0; k < j; ++k, wk += rowstride_dst) {
                uint64_t xor             = ((*wk >> j) ^ *(wk + j_rowstride_dst)) & m;
                *wk                     ^= xor << j;
                *(wk + j_rowstride_dst) ^= xor;
            }
        }
    }
}

/// Rows of all matrices are expected to fit exactly in a word (offset == 0)
/// and lay entirely inside a single block.
///
/// \note This function also works to transpose in-place.
///
/// Transpose two 64 x 64 matrix with width 1.
///
/// \param dst1 First word of destination matrix 1.
/// \param dst2 First word of destination matrix 2.
/// \param src1 First word of source matrix 1.
/// \param src2 First word of source matrix 2.
/// \param _rowstride_dst Rowstride of destination matrices.
/// \param _rowstride_src Rowstride of source matrices.
static inline void gf2_matrix_transpose_64x64_2(uint8_t *restrict dst1,
                                                uint8_t *restrict dst2,
                                                uint8_t const *restrict src1,
                                                uint8_t const *restrict src2,
                                                const uint32_t _rowstride_dst,
                                                const uint32_t _rowstride_src) {
    uint64_t m               = 0xFFFFFFFF;
    const uint32_t rowstride_dst = _rowstride_dst/8;
    const uint32_t rowstride_src = _rowstride_src/8;
    uint32_t j_rowstride_dst = rowstride_dst * 64;
    uint32_t j_rowstride_src = rowstride_src * 32;
    uint64_t *const end      = ((uint64_t *)dst1) + j_rowstride_dst;
    int j                = 32;
    uint64_t *restrict wk[2];
    uint64_t const *restrict wks[2];
    uint64_t xor [2];

    j_rowstride_dst >>= 1;
    wk[0]  = (uint64_t *)dst1;
    wk[1]  = (uint64_t *)dst2;
    wks[0] = (uint64_t *)src1;
    wks[1] = (uint64_t *)src2;

    do {
        for (int k = 0; k < j; ++k) {
            xor[0]                     = ((*wks[0] >> j) ^ *(wks[0] + j_rowstride_src)) & m;
            xor[1]                     = ((*wks[1] >> j) ^ *(wks[1] + j_rowstride_src)) & m;
            *wk[0]                     = *wks[0] ^ (xor[0] << j);
            *wk[1]                     = *wks[1] ^ (xor[1] << j);
            *(wk[0] + j_rowstride_dst) = *(wks[0] + j_rowstride_src) ^ xor[0];
            *(wk[1] + j_rowstride_dst) = *(wks[1] + j_rowstride_src) ^ xor[1];
            wk[0] += rowstride_dst;
            wk[1] += rowstride_dst;
            wks[0] += rowstride_src;
            wks[1] += rowstride_src;
        }

        wk[0] += j_rowstride_dst;
        wk[1] += j_rowstride_dst;
        wks[0] += j_rowstride_src;
        wks[1] += j_rowstride_src;

    } while (wk[0] < end);

    m ^= m << 16;
    for (j = 16; j != 0; j = j >> 1, m ^= m << j) {
        j_rowstride_dst >>= 1;
        wk[0] = (uint64_t *)dst1;
        wk[1] = (uint64_t *)dst2;

        do {

            for (int k = 0; k < j; ++k) {
                xor[0] = ((*wk[0] >> j) ^ *(wk[0] + j_rowstride_dst)) & m;
                xor[1] = ((*wk[1] >> j) ^ *(wk[1] + j_rowstride_dst)) & m;
                *wk[0] ^= xor[0] << j;
                *wk[1] ^= xor[1] << j;
                *(wk[0] + j_rowstride_dst) ^= xor[0];
                *(wk[1] + j_rowstride_dst) ^= xor[1];
                wk[0] += rowstride_dst;
                wk[1] += rowstride_dst;
            }

            wk[0] += j_rowstride_dst;
            wk[1] += j_rowstride_dst;

        } while (wk[0] < end);
    }
}

/// Rows of all matrices are expected have offset zero
/// and lay entirely inside a single block.
///
/// \note This function also works to transpose in-place.
///
/// Transpose a n x 64 matrix with width 1.
///
/// \param dst First word of destination matrix.
/// \param src First word of source matrix.
/// \param rowstride_dst Rowstride of destination matrix.
/// \param rowstride_src Rowstride of source matrix.
/// \param n Number of rows in source matrix, must be less than 64.
static inline void gf2_matrix_transpose_le64x64(uint8_t *restrict dst,
                                                uint8_t const *restrict src,
                                                const uint32_t rowstride_dst,
                                                const uint32_t rowstride_src,
                                                const uint32_t n) {
    // Preload the n input rows into level 1, using a minimum of cache lines (compact storage).
    uint64_t t[64];
    uint8_t const *restrict wks = src;
    uint64_t const m      = __LEFT_BITMASK(n);
    int k;
    for (k = 0; k < n; ++k) {
        t[k] = *(uint64_t *)wks;
        t[k] &= m;
        wks += rowstride_src;
    }
    // see https://bitbucket.org/malb/m4ri/issues/53
    for (; k < 64; ++k) { t[k] = 0; }
    if (n > 32) {
        while (k < 64) { t[k++] = 0; }
        gf2_matrix_transpose_64x64(dst, (uint8_t *)t, rowstride_dst, 8);
        return;
    }

    int log2j = _mzd_transpose_Nxjx64(t, n);
    // All output bits are now transposed, but still might need to be shifted in place.
    // What we have now is 64/j matrices of size jxj. Thus,
    // [Al]...[A1][A0], where l = 64/j - 1.
    // while the actual output is:
    // [A0]
    // [A1]
    // ...
    // [Al]
    uint8_t *restrict wk = (uint8_t *)dst;
    switch (log2j) {
        case 5: {
            const uint32_t j_rowstride_dst = 32 * rowstride_dst;
            for (uint32_t k = 0; k < 32; ++k) {
                *(uint64_t *)(wk + 0)               = t[k] & m;
                *(uint64_t *)(wk + j_rowstride_dst) = (t[k] >> 32) & m;
                wk += rowstride_dst;
            }
            break;
        }
        case 4: {
            const uint32_t j_rowstride_dst = 16 * rowstride_dst;
            for (uint32_t k = 0; k < 16; ++k) {
                wk[0]                   = t[k] & m;
                wk[j_rowstride_dst]     = (t[k] >> 16) & m;
                wk[2 * j_rowstride_dst] = (t[k] >> 32) & m;
                wk[3 * j_rowstride_dst] = (t[k] >> 48) & m;
                wk += rowstride_dst;
            }
            break;
        }
        case 3: {
            const uint32_t j_rowstride_dst = 8 * rowstride_dst;
            for (uint32_t k = 0; k < 8; ++k) {
                wk[0]                   = t[k] & m;
                wk[j_rowstride_dst]     = (t[k] >> 8) & m;
                wk[2 * j_rowstride_dst] = (t[k] >> 16) & m;
                wk[3 * j_rowstride_dst] = (t[k] >> 24) & m;
                wk[4 * j_rowstride_dst] = (t[k] >> 32) & m;
                wk[5 * j_rowstride_dst] = (t[k] >> 40) & m;
                wk[6 * j_rowstride_dst] = (t[k] >> 48) & m;
                wk[7 * j_rowstride_dst] = (t[k] >> 56) & m;
                wk += rowstride_dst;
            }
            break;
        }
        case 2: {
            const uint32_t j_rowstride_dst = 4 * rowstride_dst;
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t *restrict wk2 = wk;
                uint64_t tk            = t[k];
                for (int i = 0; i < 2; ++i) {
                    wk2[0]                   = tk & m;
                    wk2[j_rowstride_dst]     = (tk >> 4) & m;
                    wk2[2 * j_rowstride_dst] = (tk >> 8) & m;
                    wk2[3 * j_rowstride_dst] = (tk >> 12) & m;
                    wk2[4 * j_rowstride_dst] = (tk >> 16) & m;
                    wk2[5 * j_rowstride_dst] = (tk >> 20) & m;
                    wk2[6 * j_rowstride_dst] = (tk >> 24) & m;
                    wk2[7 * j_rowstride_dst] = (tk >> 28) & m;
                    wk2 += 8 * j_rowstride_dst;
                    tk >>= 32;
                }
                wk += rowstride_dst;
            }
            break;
        }
        case 1: {
            const uint32_t j_rowstride_dst = 2 * rowstride_dst;
            for (uint32_t k = 0; k < 2; ++k) {
                uint64_t *restrict wk2 = wk;
                uint64_t tk            = t[k];
                for (int i = 0; i < 8; ++i) {
                    wk2[0]                   = tk & m;
                    wk2[j_rowstride_dst]     = (tk >> 2) & m;
                    wk2[2 * j_rowstride_dst] = (tk >> 4) & m;
                    wk2[3 * j_rowstride_dst] = (tk >> 6) & m;
                    wk2 += 4 * j_rowstride_dst;
                    tk >>= 8;
                }
                wk += rowstride_dst;
            }
            break;
        }
        case 0: {
            uint64_t *restrict wk2 = wk;
            uint64_t tk            = t[0];
            for (int i = 0; i < 16; ++i) {
                wk2[0]                 = tk & m;
                wk2[rowstride_dst]     = (tk >> 1) & m;
                wk2[2 * rowstride_dst] = (tk >> 2) & m;
                wk2[3 * rowstride_dst] = (tk >> 3) & m;
                wk2 += 4 * rowstride_dst;
                tk >>= 4;
            }
            break;
        }
    }
}

/// TODO case n=1,2,3
/// Rows of all matrices are expected have offset zero
/// and lay entirely inside a single block.
///
/// \note This function also works to transpose in-place.
///
/// Transpose a 64 x n matrix with width 1.
///
/// \param dst First word of destination matrix.
/// \param src First word of source matrix.
/// \param rowstride_dst Rowstride of destination matrix.
/// \param rowstride_src Rowstride of source matrix.
/// \param n Number of columns in source matrix, must be less than 64.
static inline void gf2_matrix_transpose_64xle64(uint8_t *restrict dst,
                                                uint8_t const *restrict src,
                                                const uint32_t rowstride_dst,
                                                const uint32_t rowstride_src,
                                                const uint32_t n) {
    uint64_t t[64];
    int log2j = log2_ceil((int)n);
    const uint8_t  *restrict wks = src;
    const uint64_t mask = __LEFT_BITMASK(n);
    switch (log2j) {
        case 6: {
            gf2_matrix_transpose_64x64((uint8_t *)t, src, 8, rowstride_src);
            uint8_t *restrict wk = dst;
            for (uint32_t k = 0; k < n; ++k) {
                *(uint64_t *)wk = t[k] & mask;
                wk += rowstride_dst;
            }
            return;
        }
        case 5: {
            const uint32_t j_rowstride_src = 32 * rowstride_src;
            for (uint64_t k = 0; k < 32; ++k) {
                t[k] = ((*(uint64_t *)wks)&mask) | (*((uint64_t *)(wks + j_rowstride_src)) << 32);
                wks += rowstride_src;
            }
            break;
        }
        case 4: {
            const uint32_t j_rowstride_src = 16 * rowstride_src;
            for (int k = 0; k < 16; ++k) {
                t[k] =  *((uint64_t *)(wks + 0))                          | (*((uint64_t *)(wks +     j_rowstride_src)) << 16);
                t[k]|= (*((uint64_t *)(wks + 2 * j_rowstride_src)) << 32) | (*((uint64_t *)(wks + 3 * j_rowstride_src)) << 48);
                wks += rowstride_src;
            }
            break;
        }
        case 3: {
            const uint32_t j_rowstride_src = 8 * rowstride_src;
            uint64_t tt;
            for (int k = 0; k < 8; ++k) {
                tt   =  (*(uint64_t *)(wks + 0))                          | ((*(uint64_t *)(wks +     j_rowstride_src)) <<  8);
                t[k] = ((*(uint64_t *)(wks + 2 * j_rowstride_src)) << 16) | ((*(uint64_t *)(wks + 3 * j_rowstride_src)) << 24);
                tt  |= ((*(uint64_t *)(wks + 4 * j_rowstride_src)) << 32) | ((*(uint64_t *)(wks + 5 * j_rowstride_src)) << 40);
                t[k]|= ((*(uint64_t *)(wks + 6 * j_rowstride_src)) << 48) | ((*(uint64_t *)(wks + 7 * j_rowstride_src)) << 56);
                wks += rowstride_src;
                t[k] |= tt;
            }
            break;
        }
        case 2: {
            uint8_t const *restrict wks2 = wks + 60 * rowstride_src;
            t[0]                      = *(uint64_t *)(wks2 + 0);
            t[1]                      = *(uint64_t *)(wks2 +     rowstride_src);
            t[2]                      = *(uint64_t *)(wks2 + 2 * rowstride_src);
            t[3]                      = *(uint64_t *)(wks2 + 3 * rowstride_src);
            for (int i = 0; i < 15; ++i) {
                wks2 -= 4 * rowstride_src;
                t[0] <<= 4;
                t[1] <<= 4;
                t[2] <<= 4;
                t[3] <<= 4;
                t[0] |= wks2[0];
                t[1] |= wks2[rowstride_src];
                t[2] |= wks2[2 * rowstride_src];
                t[3] |= wks2[3 * rowstride_src];
            }
            break;
        }
        case 1: {
            wks += 62 * rowstride_src;
            t[0] = wks[0];
            t[1] = wks[rowstride_src];
            for (int i = 0; i < 31; ++i) {
                wks -= 2 * rowstride_src;
                t[0] <<= 2;
                t[1] <<= 2;
                t[0] |= wks[0];
                t[1] |= wks[rowstride_src];
            }
            break;
        }
        case 0: {
            uint64_t tt[2];
            tt[0] = wks[0];
            tt[1] = wks[rowstride_src];
            for (int i = 2; i < 64; i += 2) {
                wks += 2 * rowstride_src;
                tt[0] |= wks[0] << i;
                tt[1] |= wks[rowstride_src] << i;
            }
            *dst = tt[0] | (tt[1] << 1);
            return;
        }
    }
    int j = 1 << log2j;
    _mzd_transpose_Nxjx64(t, j);
    uint8_t *restrict wk = dst;
    for (int k = 0; k < n; ++k) {
        *(uint64_t *)wk = t[k] & mask;
        wk += rowstride_dst;
    }
}

///
/// \param dst
/// \param src
/// \param rowstride_dst
/// \param rowstride_src
/// \param n
/// \param m
static inline void gf2_matrix_transpose_le64xle64(uint8_t *restrict dst,
                                                  uint8_t const *restrict src,
                                                  const uint32_t rowstride_dst,
                                                  const uint32_t rowstride_src,
                                                  const uint32_t n,
                                                  const uint32_t m) {
    uint8_t const *restrict wks = src;
    uint64_t t[64];
    const uint64_t mask = __LEFT_BITMASK(m);
    int k;
    for (k = 0; k < n; ++k) {
        t[k] = *(uint64_t *)wks;
        t[k] &= mask;
        wks += rowstride_src;
    }
    while (k < 64) t[k++] = 0;

    gf2_matrix_transpose_64x64((uint8_t *)t, (uint8_t *)t, 8, 8);
    uint8_t *restrict wk = dst;
    for (uint32_t s = 0; s < m; ++s) {
        *(uint64_t *)wk = t[s];
        wk += rowstride_dst;
    }
}
///
/// \param fwd
/// \param fws
/// \param rowstride_dst
/// \param rowstride_src
/// \param nrows
/// \param ncols
/// \param maxsize
static inline void gf2_matrix_transpose_small(uint8_t *restrict fwd,
                                              uint8_t const *restrict fws,
                                              const uint32_t rowstride_dst,
                                              const uint32_t rowstride_src,
                                              const uint32_t nrows,
                                              const uint32_t ncols,
                                              const uint32_t maxsize) {
    assert(maxsize < 64);
    if (maxsize <= 8) {
        gf2_matrix_transpose_le8xle8(fwd, fws, rowstride_dst, rowstride_src, nrows, ncols, maxsize);
    } else if (maxsize <= 16) {
        gf2_matrix_transpose_le16xle16(fwd, fws, rowstride_dst, rowstride_src, nrows, ncols, maxsize);
    } else if (maxsize <= 32) {
        gf2_matrix_transpose_le32xle32(fwd, fws, rowstride_dst, rowstride_src, nrows, ncols);
    } else {
        gf2_matrix_transpose_le64xle64(fwd, fws, rowstride_dst, rowstride_src, nrows, ncols);
    }
}

/// TODO not implement
/// \param fwd
/// \param fws
/// \param rowstride_dst
/// \param rowstride_src
/// \param nrows
/// \param ncols
/// \param maxsize
void gf2_matrix_transpose_base(uint8_t *restrict fwd,
                               uint8_t const *restrict fws,
                               const uint32_t rowstride_dst,
                               const uint32_t rowstride_src,
                               uint32_t nrows,
                               uint32_t ncols,
                               const uint32_t maxsize) {
    // Note that this code is VERY sensitive. ANY change to _mzd_transpose can easily
    // reduce the speed for small matrices (up to 64x64) by 5 to 10%.
    if (nrows >= 64) {
        /*
         * This is an interesting #if ...
         * I recommend to investigate the number of instructions, and the clocks per instruction,
         * as function of various sizes of the matrix (most likely especially the number of columns
         * (the size of a row) will have influence; also always use multiples of 64 or even 128),
         * for both cases below.
         *
         * To measure this run for example:
         *
         * ./bench_mzd -m 10 -x 10 -p PAPI_TOT_INS,PAPI_L1_TCM,PAPI_L2_TCM mzd_transpose 32000 32000
         * ./bench_mzd -m 10 -x 100 -p PAPI_TOT_INS,PAPI_L1_TCM,PAPI_L2_TCM mzd_transpose 128 10240
         * etc (increase -x for smaller sizes to get better accuracy).
         *
         * --Carlo Wood
         */
        int js = ncols & nrows & 64;  // True if the total number of whole 64x64 matrices is odd.
        const uint32_t rowstride_64_dst      = 64 * rowstride_dst;
        uint64_t *restrict fwd_current       = (uint64_t *)fwd;
        uint64_t const *restrict fws_current = (uint64_t *)fws;
        if (js) {
            js = 1;
            gf2_matrix_transpose_64x64(fwd, fws, rowstride_dst, rowstride_src);
            if ((nrows | ncols) == 64) {
                return;
            }
            fwd_current += rowstride_64_dst;
            ++fws_current;
        }
        const uint32_t whole_64cols = ncols / 64;
        // The use of delayed and even, is to avoid calling _mzd_copy_transpose_64x64_2 twice.
        // This way it can be inlined without duplicating the amount of code that has to be loaded.
        uint64_t *restrict fwd_delayed       = NULL;
        uint64_t const *restrict fws_delayed = NULL;
        int even                         = 0;
        while (1) {
            for (int j = js; j < whole_64cols; ++j) {
                if (!even) {
                    fwd_delayed = fwd_current;
                    fws_delayed = fws_current;
                } else {
                    gf2_matrix_transpose_64x64_2(fwd_delayed, fwd_current, fws_delayed, fws_current,
                                                rowstride_dst, rowstride_src);
                }
                fwd_current += rowstride_64_dst;
                ++fws_current;
                even = !even;
            }
            nrows -= 64;
            if (ncols % 64) {
                gf2_matrix_transpose_64xle64(fwd + whole_64cols * rowstride_64_dst, fws + whole_64cols,
                                            rowstride_dst, rowstride_src, ncols % 64);
            }
            fwd += 1;
            fws += 64 * rowstride_src;
            if (nrows < 64) break;
            js          = 0;
            fws_current = fws;
            fwd_current = fwd;
        }
    }

    if (nrows == 0) {
        return;
    }

    // Transpose the remaining top rows. Now 0 < nrows < 64.
    while (ncols >= 64) {
        gf2_matrix_transpose_le64x64(fwd, fws, rowstride_dst, rowstride_src, nrows);
        ncols -= 64;
        fwd += 64 * rowstride_dst;
        fws += 1;
    }

    if (ncols == 0) {
        return ;
    }

    const uint32_t maxsize2 = MAX(nrows, ncols);
    // Transpose the remaining corner. Now both 0 < nrows < 64 and 0 < ncols < 64.
    gf2_matrix_transpose_small(fwd, fws, rowstride_dst, rowstride_src, nrows, ncols, maxsize2);
}

