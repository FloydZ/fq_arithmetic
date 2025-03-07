#!/usr/bin/env python3

q = 127
padding = True

print("const static uint8_t __gf127_lookuptable[]  __attribute__((aligned(64))) = { ")
for i in range(q):
    l = list(range(q))
    t = [str(i * a % q) for a in l]
    print("\t", ", ".join(t), end='')
    if (padding):
        print(", 0, ", end='')
    print()

print("};")
