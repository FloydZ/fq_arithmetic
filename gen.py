#!/usr/bin/env python3
""" generator script for finite field matrix and vector arithmetic """

from math import log2, ceil
from typing import List

# TODO: implement the following flags:
#   - perload_matrix)into_registers
#   - 


list_of_arguments = ["c", "a", "b"]
list_of_variables = ["t1", "t2", "t3", "t4"]


def static_vars(**kwargs):
    """ TODO """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(ctr=0)
def get_var_name():
    """ TODO """
    ret = list_of_variables[get_var_name.ctr]
    get_var_name.ctr += 1
    return ret


def generate_function_declaration(name: str, typ: str, nr_args: int) -> str:
    """generates a simple C function header
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
    :return tha smallest power of two (>= 8) s.t. >= a
    """
    tmp = [8, 16, 32, 64]
    for t in tmp:
        if t >= a:
            return t
        continue
    raise ValueError("Input to big")


def bits_to_type_str(n: int) -> str:
    """
    :param n:
    :return 
    """
    if n not in [8, 16, 32, 64]:
        raise ValueError("not in range")
    return f"uint{n}_t"


class AVX:
    @staticmethod
    def decl(regs: List[str]) -> str:
        return "__m256i " + ", ".join(regs) + ";\n"

    @staticmethod
    def load(out_var: str, in_var: str) -> str:
        return f"{out_var} = _mm256_loadu_si256((__m256i *)({in_var}));\n"

    @staticmethod
    def load_multiple(out_vars: List[str],
                      in_var: str,
                      stride: int) -> str:
        ret = ""
        for i, out_var in enumerate(out_vars):
            ret += f"{out_var} = _mm256_loadu_si256((__m256i *)({in_var} + {i}*{stride}));\n"
        return ret

    @staticmethod
    def store(out_var: str, in_var: str) -> str:
        return f"_mm256_storeu_si256((__m256i *)({out_var}), {in_var});\n"
    
    @staticmethod
    def store_multiple(out_var: str,
                       in_vars: List[str],
                       stride: int) -> str:
        ret = ""
        for i, in_var in enumerate(in_vars):
            ret += f"_mm256_storeu_si256((__m256i *)({out_var} + {i}*{stride}), {in_var});\n"
        return ret

    @staticmethod
    def xor(oreg: str, ireg1: str, ireg2: str) -> str:
        return f"{oreg} = _mm256_xor_si256({ireg1}, {ireg2});\n"

    @staticmethod
    def xor_multiple(oregs: List[str],
                     iregs1: List[str], 
                     iregs2: List[str]) -> str:
        assert len(oregs) == len(iregs1) == len(iregs2)
        ret = ""
        for i in range(len(oregs)):
            oreg, ireg1, ireg2 = oregs[i], iregs1[i], iregs2[i]
            ret += f"{oreg} = _mm256_xor_si256({ireg1}, {ireg2});\n"
        return ret


class Shape:
    """
    TODO
    """

    SUB_LIMB_LIMIT:int = 4

    def __init__(self, n: int,
                 q: int,
                 mu: int,
                 simd_width: int,
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
        self.q, self.mu, self.n, self.simd_width, self.padding =\
            q, mu, n, simd_width, padding
        self.tail = not padding

        # NOTE: well, actually thatsnot 100% correct. It could be that only 
        # 7 bits are used, so technically we need to round here
        self.bits_q = ceil(log2(q)) 
        self.bits_q_mu = ceil(log2(q**mu))
        # which is done here
        self.bits_limb_q = ceil_power_of_2(self.bits_q)
        self.bits_limb_q_mu = ceil_power_of_2(self.bits_q_mu)

        self.use_sub_limbs_q = self.bits_q <= Shape.SUB_LIMB_LIMIT
        self.use_sub_limbs_q_mu = self.bits_q_mu <= Shape.SUB_LIMB_LIMIT

        self.q_mu_per_limb = self.bits_limb_q_mu//self.bits_q_mu if self.use_sub_limbs_q_mu else 1
        self.q_mu_per_simd = (self.simd_width//self.bits_limb_q_mu) * self.q_mu_per_limb
        self.limb_per_simd_q_mu = self.simd_width//self.bits_limb_q_mu

        tmp = (((n*self.bits_limb_q_mu + self.simd_width - 1) // self.simd_width)) * self.q_mu_per_simd
        self.internal_n = n if not padding else tmp

        self.number_limbs_q_mu = self.internal_n // self.q_mu_per_limb
        self.number_simd_q_mu = self.internal_n // self.q_mu_per_simd

        self.bits = self.number_limbs_q_mu * self.bits_limb_q_mu

        self.limb_type_str = bits_to_type_str(self.bits_limb_q_mu)
        self.simd_type_str = "__m256i" # TODO not generic


class Vector(Shape):
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
                 simd_width: int = 32,
                 padding=False) -> None:
        """
        :param n: length of the vector
        :param q: base prime
        :param mu: extension degree
        :param simd_width: TODO not sure whats correct here. Either:
            - u8x32 as a enum (u8x16 for neon)
            - 256: simply as a number (128 for  neon)
        :param padding: if true the script assumes that the allocated length of
            the vector is a multiple (in terms of gf elements) of `simd_width`.
        """
        super().__init__(n, q, mu, simd_width, padding)
        # print(self.__dict__)


    def gen_add(self) -> str:
        """
        """
        SIMD = AVX
        total_number_of_simd = 2*self.number_limbs_q_mu 
        in1_simd_names = [get_var_name() for _ in range(self.number_simd_q_mu)]
        in2_simd_names = [get_var_name() for _ in range(self.number_simd_q_mu)]

        ret = generate_function_declaration("add", self.limb_type_str, 3)
        ret += "{\n"
        ret += SIMD.decl(in1_simd_names + in2_simd_names)
        ret += SIMD.load_multiple(in1_simd_names, "a", self.limb_per_simd_q_mu)
        ret += SIMD.load_multiple(in2_simd_names, "b", self.limb_per_simd_q_mu)
        ret += SIMD.xor_multiple(in1_simd_names, in1_simd_names, in2_simd_names)
        ret += SIMD.store_multiple("c", in1_simd_names, self.limb_per_simd_q_mu)
        

        ret += "}\n"
        return ret


class Matrix(Vector):
    def __init__(self, q: int,
                 mu: int, 
                 n: int,
                 m: int,
                 k: int,
                 simd_width: int = 256,
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
        pass


if __name__ == "__main__":
    #v = Vector(31, 127, 2, 256, padding=True)
    v = Vector(31, 2, 1, 256, padding=True)
    print("""
#include <immintrin.h>
#include <stdint.h>
""")
    print(v.gen_add())
