#include <stdbool.h>
#include "arith.h"


bool test1() {
	gf2 a = 1, b = 1;
	if (gf2_add(a, b)) { return false; }

	return true;
}

int main() {
	if (!test1()) { return -1; }

	return 0;
}
