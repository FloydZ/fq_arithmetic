#!/usr/bin/env python3
""" generator script for finite field matrix and vector arithmetic """

from math import log2, ceil
from typing import List

# TODO: implement the following flags:
#   - preload_matrix_into_registers (load everything into registers and then do the math)
#   - instead of instrinsics, emit actual instructions
#   - the shape class exports to many internal values. make them private
#   - make the access to the `list_of_arguments` generic: like the name of the out param is always the same
#   - make the `generate_function_declaration` more abstract, support different type for each argument.


list_of_arguments = ["c", "a", "b"]
list_of_variables = ["t" + str(i) for i in range(100)]


def static_vars(**kwargs):
    """ helper function, injecting static variables into a function """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(ctr=0)
def get_var_name():
    """
    :return the next free name of a variable.
    """
    ret = list_of_variables[get_var_name.ctr]
    get_var_name.ctr += 1
    return ret


def generate_function_declaration(name: str, typ: str, nr_args: int) -> str:
    """generates a simple C function header
    :param name: name of the function 
    :param typ: type of the input arguments
    :param nr_args: number of input arguments
    :return `void {name}({typ} *arg1);`
    """
    assert nr_args <= len(list_of_arguments)
    ret = f"void {name}(\n"
    for i in range(nr_args):
        tmp = list_of_arguments[i]
        ret += f"{typ} *{tmp}, "
        # if i < (nr_args - 1): ret += ","
        ret += "\n"
    ret += "const size_t n\n"
    ret += ")\n"
    return ret


def ceil_mutliple_of(a: int, n: int) -> int:
    """
    returns the smallest x >= a, s.t. x%n == 0
    :param a: number to ceil
    :param n: muliple
    :return ((a + n - 1) // n) * n
    """
    return ((a + n - 1) // n) * n


def ceil_power_of_2(a: int) -> int:
    """
    :param a: input value
    :return the smallest power of two (>= 8) s.t. >= a
    """
    tmp = [8, 16, 32, 64]
    for t in tmp:
        if t >= a:
            return t
        continue
    raise ValueError("Input to big")


def bits_to_type_str(n: int) -> str:
    """
    :param n: integer in [8, 16, 32, 64]
    :return: `uint{n}_t`
    """
    if n not in [8, 16, 32, 64]:
        raise ValueError("not in range")
    return f"uint{n}_t"


class SIMD:
    def __init__(self, q:int) -> None:
        self.register_name = ""
        self.q_str = str(q)
        self.q = q
        self.aligned_instructions = False
        self.n = ceil_power_of_2(ceil(log2(q)))
        if self.n not in [8, 16, 32, 64]:
            raise ValueError("not in range")

    def decl(self, regs: List[str]) -> str:
        """
        :param regs: list of registers the declare 
        :return a string containing the register declaration
        """
        return self.register_name + " " + ", ".join(regs) + ";\n"

    def load(self,
             out_var: str,
             in_var: str) -> str:
        raise NotImplemented
    
    def load_multiple(self,
                      out_vars: List[str],
                      in_var: str,
                      stride: int) -> str:
        """
        :param out_vars: list of register variables to store the result
        :param in_var: name of the memory variable
        :param stride: number of Fq limbs between two loads. NOTE: not bytes
        """
        ret = ""
        for i, out_var in enumerate(out_vars):
            ret += self.load(out_var, f"{in_var} + {i}*{stride}")
        return ret

    def store(self,
              out_var: str,
              in_var: str) -> str:
        raise NotImplemented

    def store_multiple(self,
                       out_var: str,
                       in_vars: List[str],
                       stride: int) -> str:
        """
        :param out_var: name of variable containing memory location
        :param in_vars: names of registers to store
        :param stride: number of Fq limbs between two stores. NOTE: not bytes.
        """
        ret = ""
        for i, in_var in enumerate(in_vars):
            ret += self.store(f"{out_var} + {i}*{stride}", in_var)
        return ret

    def set1(self,
            out_var: str,
            in_var: str) -> str:
        raise NotImplemented

    def set1_multiple(self,
                     out_vars: List[str],
                     in_vars: List[str]) -> str:
        """
        :param out_vars: list of register variables to store the result
        :param in_var: list of variables to set
        :param stride: number of Fq limbs between two loads. NOTE: not bytes
        """
        ret = ""
        for i, out_var in enumerate(out_vars):
            ret += self.expand(out_var, in_vars[i])
        return ret
    
    def expand(self,
               out_var: str,
               in_var: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: name of the variable containing the memory location
        """
        return f"{out_var} = gf{self.q_str}v_expand_u{self.n}({in_var});\n"

    def expand_multiple(self,
                        out_vars: List[str],
                        in_var: str,
                        stride: int) -> str:
        """
        :param out_vars: list of register variables to store the result
        :param in_var: name of the memory variable
        :param stride: number of Fq limbs between two loads. NOTE: not bytes
        """
        ret = ""
        for i, out_var in enumerate(out_vars):
            ret += self.expand(out_var, f"{in_var} + {i}*{stride}")
        return ret

    def add(self,
            oreg: str,
            ireg1: str,
            ireg2: str) -> str:
        """
        NOTE: this is the Fq addition
        :param oreg: output Register
        :param ireg1: first input register 
        :param ireg2: second input register
        :return: a string containing the call to the vectorized addition function
        """
        return f"{oreg} = gf{self.q_str}v_add_u{self.n}({ireg1}, {ireg2});\n"
    

    def add_multiple(self,
                     oregs: List[str],
                     iregs1: List[str], 
                     iregs2: List[str]) -> str:
        """
        :param oreg: output Registers
        :param ireg1: first input registers
        :param ireg2: second input registers
        :return: a string containing the call to multiple vectorized addition function
        """
        assert len(oregs) == len(iregs1) == len(iregs2)
        ret = ""
        for i in range(len(oregs)):
            oreg, ireg1, ireg2 = oregs[i], iregs1[i], iregs2[i]
            ret += self.add(oreg, ireg1, ireg2)
        return ret
    
    def mul(self,
            oreg: str,
            ireg1: str,
            ireg2: str) -> str:
        """
        NOTE: this is the Fq mul
        :param oreg: output Register
        :param ireg1: first input register 
        :param ireg2: second input register
        :return: a string containing the call to the vectorized addition function
        """
        return f"{oreg} = gf{self.q_str}v_mul_u{self.n}({ireg1}, {ireg2});\n"

    def mul_multiple(self,
                     oregs: List[str],
                     iregs1: List[str], 
                     iregs2: List[str]) -> str:
        """
        :param oreg: output Registers
        :param ireg1: first input registers
        :param ireg2: second input registers
        :return: a string containing the call to multiple vectorized addition function
        """
        assert len(oregs) == len(iregs1) == len(iregs2)
        ret = ""
        for i in range(len(oregs)):
            oreg, ireg1, ireg2 = oregs[i], iregs1[i], iregs2[i]
            ret += self.add(oreg, ireg1, ireg2)
        return ret


class SSE(SIMD):
    def __init__(self, q: int) -> None:
        super().__init__(q)
        self.register_width = 128
        self.register_name = "__m128i"

    def load(self,
             out_var: str,
             in_var: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: name of the variable containing the memory location
        """
        if self.aligned_instructions:
            return f"{out_var} = _mm_load_si128((__m128i *)({in_var}));\n"
        return f"{out_var} = _mm_loadu_si128((__m128i *)({in_var}));\n"

    def store(self,
              out_var: str,
              in_var: str) -> str:
        """
        :param out_var: variable name containing pointer to memory
        :param in_var: register variable
        """
        if self.aligned_instructions:
            return f"_mm_store_si128((__m128i *)({out_var}), {in_var});\n"
        return f"_mm_storeu_si128((__m128i *)({out_var}), {in_var});\n"
    
    def set1(self,
             out_var: str,
             in_var: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: value to set
        """
        return f"{out_var} = _mm_set1_epi{self.n}({in_var});\n"
    

class AVX(SIMD):
    def __init__(self, q:int) -> None:
        super().__init__(q)
        self.register_width = 256
        self.register_name = "__m256i"

    def load(self,
             out_var: str,
             in_var: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: name of the variable containing the memory location
        """
        if self.aligned_instructions:
            return f"{out_var} = _mm256_load_si256((__m256i *)({in_var}));\n"
        return f"{out_var} = _mm256_loadu_si256((__m256i *)({in_var}));\n"

    def store(self,
              out_var: str,
              in_var: str) -> str:
        """
        :param out_var: variable name containing pointer to memory
        :param in_var: register variable
        """
        if self.aligned_instructions:
            return f"_mm256_store_si256((__m256i *)({out_var}), {in_var});\n"
        return f"_mm256_storeu_si256((__m256i *)({out_var}), {in_var});\n"
    
    def set1(self,
             out_var: str,
             in_var: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: value to set
        """
        return f"{out_var} = _mm256_set1_epi{self.n}({in_var});\n"
    

class AVX512(AVX):
    def __init__(self, q: int) -> None:
        super().__init__(q)
        self.register_width = 512
        self.register_name = "__m512i"
    
    def load(self,
             out_var: str,
             in_var: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: name of the variable containing the memory location
        """
        if self.aligned_instructions:
            return f"{out_var} = _mm512_load_si512((__m512i *)({in_var}));\n"
        return f"{out_var} = _mm512_loadu_si512((__m512i *)({in_var}));\n"

    def store(self,
              out_var: str,
              in_var: str) -> str:
        """
        :param out_var: variable name containing pointer to memory
        :param in_var: register variable
        """
        if self.aligned_instructions:
            return f"_mm512_store_si512((__m512i *)({out_var}), {in_var});\n"
        return f"_mm512_storeu_si512((__m512i *)({out_var}), {in_var});\n"
    
    def set1(self,
             out_var: str,
             in_var: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: value to set
        """
        return f"{out_var} = _mm512_set1_epi{self.n}({in_var});\n"


class NEON(SIMD):
    def __init__(self, q: int) -> None:
        super().__init__(q)
        self.register_width = 128
        t =  self.register_width // self.n
        self.register_name = f"uint{self.n}x{t}_t"

    def load(self,
             out_var: str,
             in_var: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: name of the variable containing the memory location
        """
        return f"{out_var} = vld1q_u{self.n}(({self.register_name} *)({in_var}));\n"

    def store(self,
              out_var: str,
              in_var: str) -> str:
        """
        :param out_var: variable name containing pointer to memory
        :param in_var: register variable
        """
        return f"vst1q_u(({self.register_name} *)({out_var}), {in_var});\n"
    
    def set1(self,
             out_var: str,
             in_var: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: value to set
        """
        return f"{out_var} = vdupq_n_u{self.n}({in_var});\n"


class Shuffler:
    pass


class Shape:
    """
    TODO describe
    """

    SUB_LIMB_LIMIT:int = 4

    def __init__(self, n: int,
                 q: int,
                 mu: int,
                 simd,
                 padding: bool = False) -> None:
        """
        Nomenclature: Example of a SSE/NEON Register and F2
                 ┌───────┬───────┬───────┬───────┐
        Simdlimb │       │       │       │       │
                 └───────┴───────┴───────┴───────┘
                 0       31     63      95     127 Bits
                 ┌───────┐
        Limb     │       │
                 └───────┘
                 0 1   Example:                        
                 ┌─┐
        Sublimb  │ │        F2                         
                 └─┘
                 0     7
                 ┌─────┐
        Sublimb  │     │    F127                       
                 └─────┘

        :param n: length of the vector in fq^mu elements
        :param q: base prime
        :param mu: extension degree
        :param simd_width: TODO
        :param padding: TODO
        """
        self.q, self.mu, self.n, self.simd, self.padding =\
            q, mu, n, simd, padding
        self.tail = not padding
        self.simd_width = self.simd.register_width

        # TODO, decide which of the values should be exported, like add `_` everywhere

        # NOTE: well, actually thatsnot 100% correct. It could be that only 
        # 7 bits are used, so technically we need to round here
        self.bits_q = ceil(log2(q))
        self.bits_q_mu = ceil(log2(q**mu))
        # which is done here
        self.bits_limb_q = ceil_power_of_2(self.bits_q)
        self.bits_limb_q_mu = ceil_power_of_2(self.bits_q_mu)
        self.use_sub_limbs_q = self.bits_q <= Shape.SUB_LIMB_LIMIT
        self.use_sub_limbs_q_mu = self.bits_q_mu <= Shape.SUB_LIMB_LIMIT

        self.q_per_limb = self.bits_limb_q//self.bits_q if self.use_sub_limbs_q else 1
        self.q_mu_per_limb = self.bits_limb_q_mu//self.bits_q_mu if self.use_sub_limbs_q_mu else 1
        self.q_per_simd = (self.simd_width//self.bits_limb_q) * self.q_per_limb
        self.q_mu_per_simd = (self.simd_width//self.bits_limb_q_mu) * self.q_mu_per_limb
        self.limb_per_simd_q = self.simd_width//self.bits_limb_q
        self.limb_per_simd_q_mu = self.simd_width//self.bits_limb_q_mu

        scale = self.bits_q_mu if self.use_sub_limbs_q_mu else self.bits_limb_q_mu
        tmp = (((n*scale + self.simd_width - 1) // self.simd_width)) * self.q_mu_per_simd
        self.internal_n = n if not padding else tmp

        # number of limbs to represent `internal_n` fq^mu elements
        self.number_limbs_q_mu = self.internal_n // self.q_mu_per_limb
        # number of simds to represent `internal_n` fq^mu elements
        self.number_simd_q_mu = self.internal_n // self.q_mu_per_simd
        # total number of bits needed
        self.bits = self.number_limbs_q_mu * self.bits_limb_q_mu

        # something like `uint8_t`, ..., `uint64_t`
        self.limb_type_str = bits_to_type_str(self.bits_limb_q_mu)
        # something like `__m256i` or `uint8x8_t`
        self.simd_type_str = self.simd.register_name
        
        # number of fq_mu limbs between two consecutive expand calls. It is 
        # basically the stride
        self.expand_number_limbs = self.limb_per_simd_q_mu // self.q_per_limb if self.use_sub_limbs_q else self.q_mu_per_simd 


class Vector:
    """
    Vector generation class
    Assumes the existance of the following functions:
        - z = gf256_add_u256(x, y)
        - z = gf256_mul_u256(x, y)
        - TODO gf_q extensions to gf_{q^mu}
    """

    def __init__(self, n: int,
                 q: int,
                 mu: int,
                 simd,
                 padding=False) -> None:
        """
        :param n: length of the vector
        :param q: base prime
        :param mu: extension degree
        :param simd_width: SIMD type, like AVX, AVX512, NEON
        :param padding: if true the script assumes that the allocated length of
            the vector is a multiple (in terms of gf elements) of `simd_width`.
        """
        self.q, self.mu, self.n, self.simd, self.padding =\
            q, mu, n, simd, padding

    def gen_add(self, fn_name: str = "vector_add") -> str:
        """
        :param fn_name: name of the function to emit. Needed, as this funciton
            is called by the `Matrix` interface
        :return a string contain the vector addition
        """
        s = Shape(self.n, self.q, self.mu, self.simd, self.padding)
        in1_simd_names = [get_var_name() for _ in range(s.number_simd_q_mu)]
        in2_simd_names = [get_var_name() for _ in range(s.number_simd_q_mu)]

        ret = generate_function_declaration(fn_name, s.limb_type_str, 3)
        ret += "{\n"
        ret += self.simd.decl(in1_simd_names + in2_simd_names)
        ret += self.simd.load_multiple(in1_simd_names, "a", s.limb_per_simd_q_mu)
        ret += self.simd.load_multiple(in2_simd_names, "b", s.limb_per_simd_q_mu)
        ret += self.simd.add_multiple(in1_simd_names, in1_simd_names, in2_simd_names)
        ret += self.simd.store_multiple("c", in1_simd_names, s.limb_per_simd_q_mu)
        ret += "}\n"
        return ret

    def gen_extend(self, fn_name: str = "vector_extend") -> str:
        """
        :param fn_name: name of the function to emit. Needed, as this funciton
            is called by the `Matrix` interface
        :return a string contain the vector extention
        """
        s = Shape(self.n, self.q, self.mu, self.simd, self.padding)
        in1_simd_names = [get_var_name() for _ in range(s.number_simd_q_mu)]

        ret = generate_function_declaration(fn_name, s.limb_type_str, 2)
        ret += "{\n"
        ret += self.simd.decl(in1_simd_names)
        ret += self.simd.expand_multiple(in1_simd_names, "a", s.expand_number_limbs)
        ret += self.simd.store_multiple("c", in1_simd_names, s.limb_per_simd_q_mu)
        ret += "}\n"
        return ret


class Matrix:
    def __init__(self, 
                 n: int,
                 m: int,
                 k: int,
                 q: int,
                 mu: int, 
                 simd: SIMD,
                 col_major=True,
                 padding=False) -> None:
        """
             k           m           k    
         ┌───────┐   ┌───────┐   ┌───────┐
         │       │   │       │   │       │
        n│   C   │ = │   A   │n m│   B   │
         │       │   │       │   │       │
         └───────┘   └───────┘   └───────┘

        :param q: base prime
        :param mu: extension degree
        :param n: number of rows in A
        :param m: number of cols in A
        :param k: number of cols in B
        :param simd_width: TODO 
        :param col_major: if true, the matrices are column major else row major
        :param padding: if true the script assumes that the allocated length of
        each col (or row if row major) is a multiple (in terms of gf elements)
        of `simd_width`.
        """
        self.q, self.mu, self.n, self.m, self.k, self.simd, self.padding, self.col_major =\
            q, mu, n, m, k, simd, padding, col_major

    def gen_add(self, fn_name: str = "matrix_add") -> str:
        """
        :return a string contain the matrix addition
        """
        assert self.k == self.m
        v = Vector(self.m*self.n, self.q, self.mu, self.simd, self.padding)
        return v.gen_add(fn_name=fn_name)

    def gen_matrix_vector_mul(self, fn_name: str = "matrix_vector_mul") -> str:
        """
             k           m         1    
         ┌───────┐   ┌───────┐     ┌┐
         │       │   │       │     ││
        n│   C   │ = │   A   │n · m││B
         │       │   │       │     ││
         └───────┘   └───────┘     └┘
        :return a string contain the matrix vector mul
        """
        A_s = Shape(self.n, self.q, self.mu, self.simd, self.padding)
        B_s = Shape(self.m, self.q, self.mu, self.simd, self.padding)

        # load B into mem
        A_simd_names = [get_var_name() for _ in range(A_s.number_simd_q_mu)]
        B_simd_name = get_var_name()

        ret = generate_function_declaration(fn_name, A_s.limb_type_str, 3)
        ret += "{\n"
        ret += A_s.simd.decl(A_simd_names)
        ret += B_s.simd.decl([B_simd_name])

        for i in range(self.m):
            # load B into register
            ret += self.simd.set1(B_simd_name, f"b[{i}]")

            # load A into registers
            ret += A_s.simd.load_multiple(A_simd_names, "a", A_s.limb_per_simd_q_mu)

            # mul each 
            for r in A_simd_names:
                ret += self.simd.mul(r, r, B_simd_name)

            # store into C
            ret += self.simd.store_multiple("c", A_simd_names, A_s.limb_per_simd_q_mu)

            # update the pointers 
            ret += f"a += {A_s.limb_per_simd_q_mu};\n"
            ret += f"c += {A_s.limb_per_simd_q_mu};\n"
        ret += "}\n"
        return ret

    def gen_matrix_matrix_mul(self, fn_name: str = "matrix_vector_mul") -> str:
        """
             k           m           k    
         ┌───────┐   ┌───────┐     ┌───────┐
         │       │   │       │     │       │
        n│   C   │ = │   A   │n · m│   B   │
         │       │   │       │     │       │
         └───────┘   └───────┘     └───────┘
        :return a string contain the matrix vector mul
        """
        A_s = Shape(self.n, self.q, self.mu, self.simd, self.padding)
        B_s = Shape(self.m, self.q, self.mu, self.simd, self.padding)

        # load B into mem
        A_simd_names = [get_var_name() for _ in range(A_s.number_simd_q_mu)]
        B_simd_name = get_var_name()

        ret = generate_function_declaration(fn_name, A_s.limb_type_str, 3)
        ret += "{\n"
        ret += A_s.simd.decl(A_simd_names)
        ret += B_s.simd.decl([B_simd_name])

        for i in range(self.m):
            # TODO
            pass
        ret += "}\n"
        return ret


if __name__ == "__main__":
    #simd = AVX(16)
    #v = Vector(31, 16, 3, simd, padding=True)
    #print(v.gen_extend())

    #simd = AVX(127)
    #v = Vector(31, 127, 2, simd, padding=True)
    #print(v.gen_extend())

    #simd = AVX(2)
    #v = Vector(33, 2, 8, simd, padding=True)
    #print(v.gen_extend())

    #simd = AVX(127)
    #v = Matrix(32, 32, 32, 127, 1, simd, padding=True)
    simd = AVX(127)
    v = Matrix(32, 32, 32, 127, 1, simd, padding=True)
    print("""
#include <immintrin.h>
#include <stdint.h>
""")
    #print(v.gen_add())
    #print(v.gen_extend())
    #print(v.gen_matrix_vector_mul())
    print(v.gen_matrix_matrix_mul())
