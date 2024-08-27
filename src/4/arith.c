#include <stdbool.h>
#include "arith.h"


bool test1() {
	if (gf4_add(0, 1) != 1) { return false; }
	if (gf4_add(0, 2) != 2) { return false; }
	if (gf4_add(0, 3) != 3) { return false; }
	if (gf4_add(0, 0) != 0) { return false; }

    if (gf4_mul(0, 0) != 0) { return false; }
    if (gf4_mul(0, 1) != 0) { return false; }
    if (gf4_mul(0, 2) != 0) { return false; }
    if (gf4_mul(0, 3) != 0) { return false; }
	return true;
}

int main() {
	if (!test1()) { return -1; }

	return 0;
}
