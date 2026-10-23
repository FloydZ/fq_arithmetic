#pragma once

/// The NEON (PMULL) implementation of GF(2^256) has been merged into
/// "arith.h" (see `gf2to256_mul_neon`, `gf2to256v_mul_neon_v2` and the
/// canonical `gf2to256v_{add,sub,mul}_u128`). This header is only kept for
/// backwards compatibility.
#include "arith.h"
