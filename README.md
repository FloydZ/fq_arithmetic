# fq_arithmetic

Efficient implementation for a few primes and prime powers. Currently, the 
following fields are supported:
- 2
- 4
- 16
- 16**2
- 16**3
- 2**128

All field implementations do export the typename `ff_t` which is the minimal
datatype, which is capable to hold a single field element.
In addition the basic arithmetic operations `add, sub, inv, mul` are provided.
Each of those operations take a `ff_t` elements. Additionally, special `add_T,
sub_T, mul_T` functions are providied which computes vectorized values. Meaning
`_add_T(a, b)` computes (for `T=uint64_t` and `p=2`) 64 additions in parallel.

Finally Matrix operations are also provided.
