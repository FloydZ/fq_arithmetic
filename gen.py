#!/usr/bin/env python3
""" generator script for finite field matrix and vector arithmetic """

from math import log2, ceil
from typing import List, Union, Tuple

# TODO: implement the following flags:
#   - preload_matrix_into_registers (load everything into registers and then do the math)
# SHAPE class:
#   - the shape class exports to many internal values. make them private
# TESTS:
#   - well, start adding tests. The problem is: I dont want to have string comparisons in the tests. They would be simply too naiv. 


# NOTE: this will not be implemented, otherwise we needed to keep track of all 
# kinds of register allocation, for example memory loads needed to always use 
# the rcx register as a memory offset in a loop. 
# global variable/configuration:
# if this value is true, the code generators will emit pure assembly.
# if this value is false, the code generators will emit C intrinsics
# use_assembly = True

list_of_arguments = ["a", "b", "c", "d", "e", "f", "g"]
list_of_variables = ["t" + str(i) for i in range(100)]

list_of_arguments_asm = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
list_of_variables_asm_x86= ["r10", "r11", "r12", "r13", "r14", "r15"]
list_of_variables_asm_x86_avx2 = ["ymm0", "ymm1", "ymm2", "ymm3", "ymm4"]
list_of_variables_asm_x86_avx512 = ["zmm0", "zmm1", "zmm2", "zmm3", "zmm4"]


list_of_variables_asm_armv7_32 = [f"r{i}" for i in range(0, 16)] # 32bit regs
list_of_variables_asm_armv7_64 = [f"d{i}" for i in range(0, 32)] # 64bit regs
list_of_variables_asm_aarch64_32 = [f"w{i}" for i in range(0, 31)] # 32bit regs
list_of_variables_asm_aarch64_64 = [f"x{i}" for i in range(0, 31)] # 64bit regs
list_of_variables_asm_aarch64_neon = [f"v{i}" for i in range(0, 32)] # 128bit regs

# NOTE:
global_variables = [] # TODO: add to main code generator


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


def generate_function_declaration(name: str, 
                                  typ: Union[str, List[str]], 
                                  nr_args: int) -> Tuple[str, List[str]]:
    """generates a simple C function header of the form:
        {name}({typ} *out_var, {typ} *in_var1, {typ} *in_var2,..., {typ} *in_var_n-2, const size_t t)
    :param name: name of the function 
    :param typ: type of the input arguments, either a single string/type, than 
        all input/output arguments will have the same type. Or a list of types,
        than the arguments can have a different 
    :param nr_args: number of input arguments
    :return [
        `void {name}({typ} *arg1, {typ} *arg2);`, 
        ["arg1", "arg2"]
    ]
    """
    assert nr_args <= len(list_of_arguments)
    if isinstance(typ, List):
        assert len(typ) == nr_args

    if isinstance(typ, str):
        typ = [typ] *nr_args

    ret = f"void {name}(\n"
    for i in range(nr_args):
        t = typ[i]
        tmp = list_of_arguments[i]
        ret += f"{t} *{tmp}, "
        # if i < (nr_args - 1): ret += ","
        ret += "\n"
    ret += "const size_t n\n"
    ret += ")\n"
    return ret, list_of_arguments[:nr_args]


def ceil_mutliple_of(a: int, n: int) -> int:
    """
    returns the smallest x >= a, s.t. x%n == 0
    :param a: number to ceil
    :param n: muliple
    :return ((a + n - 1) // n) * n
    """
    return ((a + n - 1) // n) * n


def ceil_power_of_2_for_type(a: int) -> int:
    """ NOTE: always returns atleast 8
    :param a: input value
    :return the smallest power of two (>= 8) which is bigger than `a`
    """
    tmp = [8, 16, 32, 64]
    for t in tmp:
        if t >= a:
            return t
        continue
    raise ValueError("Input to big")

def ceil_power_of_2(a: int) -> int:
    """
    :param a: input value
    :return the smallest power of two which is bigger than `a`
    """
    tmp = [1, 2, 4, 8, 16, 32, 64]
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
    if n not in [1, 2, 4, 8, 16, 32, 64]:
        raise ValueError("not in range")
    if n < 8:
        n = 8
    return f"uint{n}_t"


class SIMD:
    def __init__(self, q:int) -> None:
        self.register_name = ""
        self.q_str = str(q)
        self.q = q
        self.bits_q = ceil_power_of_2(ceil(log2(q)))
        self.aligned_instructions = False
        self.n = ceil_power_of_2_for_type(ceil(log2(q)))
        if self.n not in [1, 2, 4, 8, 16, 32, 64]:
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
        del out_var; del in_var;
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
        del out_var; del in_var;
        raise NotImplementedError

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
        del out_var; del in_var;
        raise NotImplementedError

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

    def shuffle(self, p: List[int],
                o: str,
                a: str) -> str:
        del p; del o; del a;
        raise NotImplementedError()

    def shuffle_multiple(self, p: List[int],
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
            ret += self.shuffle(p, out_var, f"{in_var} + {i}*{stride}")
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

    def setX(self,
             out_var: str,
             in_var: str,
             n: int,
             p: int) -> str:
        """
        :param p: a permutation of length 32/16/8/4/2
        :param o: output register 
        :param a: input register 
        :return TODO
        """
        del out_var; del in_var; del n; del p;
        raise NotImplementedError
    
    def shuffle(self, p: List[int],
                o: str,
                a: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: value to set
        """
        ret = ""
        t = len(p)
        assert t in [2, 4, 8, 16, 32]
        if t == 2:
            if p[0] == 0:
                # emit nothing
                return ""
            return f"{o} = ({self.register_name})_mm_shuffle_pd((__m128){a}, (__m128){a}, 0b01);"
        if t == 4:
            imm = "0b" + "".join(['{0:02b}'.format(l) for l in p[::-1]])
            return f"{o} = _mm_shuffle_pd((__m128){a}, {imm});\n"
        if t == 8:
            raise NotImplementedError()
        if t == 16:
            b = get_var_name()
            c = ", ".join([str(pp) for pp in p])
            s = f"const {self.register_name} {b} = _mm256_set1_epi8({c});\n"
            #global global_variables
            global_variables.append(s)
            return f"{o} = _mm256_shuffle_epi8({a}, {b});\n"

        return ret;
    
    def hxor(self, p: int,
             o: str,
             a: str) -> str:
        """ computes the horizonatl xor of all limbs within the given register.
        :param p: number of limbs to xor up
        :param o: output register 
        :param a: input register 
        :return TODO
        """
        if p not in [2, 4, 8, 16, # normal limb operations
                     32, 64, 128]: # sub limb operations
            raise ValueError
        if p == 2:
            # return uint64_t 
            return f"{0} = _mm_extract_epi64({a}, 0) ^ _mm_extract_epi64({a}, 1);\n"
        ret = f"{a} ^= (__m128i)_mm_shuffle_pd((__m128){a}, (__m128){a}, 0b01);\n"
        if p >= 4:
            ret += f"{a} ^= _mm_shuffle_epi32({a}, {a}, 0b01001110);\n"
            if p == 4:
                return ret + f"{o} = _mm_extract_epi32({a}, 0);\n"
        if p >= 8:
            raise NotImplemented

        if p >= 16:
            ret += self.shuffle([1, 0, 2, 3, 4, 5, 6, 7], "a", "a")
            if p == 4:
                return ret + f"{o} = _mm_extract_epi8({a}, 0);\n"
        # TODO sublimb xor
        raise NotImplemented

    @staticmethod 
    def test():
        a = SSE(16)
        print(a.shuffle([0,1,2,3], "a", "o"))
        print(a.shuffle([2,3,0,1], "a", "o"))
    

class AVX(SIMD):
    def __init__(self, q:int) -> None:
        super().__init__(q)
        self.register_width = 256
        self.register_name = "__m256i"
    
    @staticmethod
    def __check_lanes128(p: List[int]) -> bool:
        """
        :param p: a permutation of length 32/16/8/4/2
        :return true of the permutation is accross the lanes
        """
        t = len(p)
        assert t in [2, 4, 8, 16, 32]

        m = t // 2
        for i in range(m):
            if (p[i] >= m) or p[i + m] < m:
                return True 
        return False

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
    
    def setX(self,
             o: str,
             a: str,
             n: int) -> str:
        """
        :param o: name of the output variable (register)
        :param a: name of pointer to read n elements from 
        :param n: number of elements to extend
        """
        if self.bits_q < 8:
            # sub limb case: 
            nr_of_bits = self.bits_q * n
            load_type = bits_to_type_str(nr_of_bits)
            # integer type
            t1 = get_var_name()
            assert nr_of_bits <= 64

            # output sublimb type within a register
            out_sublimb_type_simd = 256//n

            ret = f"uint64_t {t1} = *(({load_type} *)({a}));\n"
            mask_ = "0b" + ("0"*(nr_of_bits - self.bits_q)  + "1"*self.bits_q)*n
            mask = hex(int(mask_, 2))
            ret += f"{t1} = _pdep_u64({t1}, {mask});\n"
            ret += f"_mm256_cvtepu{nr_of_bits}_epi{out_sublimb_type_simd}(t3);"
            # TODO not finished
 
            return ret
        return f""
    
    def shuffle(self, p: List[int],
                o: str,
                a: str) -> str:
        """ TODO implement sub byte permutation
        :param p: a permutation of length 32/16/8/4/2
        :param o: output register 
        :param a: input register 
        :return a string which applies the permutation to the input register.
            NOTE: maybe global variables are emitted.
        """
        ret = ""
        t = len(p)
        assert t in [2, 4, 8, 16, 32]
        if AVX.__check_lanes128(p):
            if t == 2:
                imm = "0b" + "000".join(['{0:02b}'.format(l) for l in p[::-1]])
                return f"{o} = _mm256_permute2x128_si256({a}, {a}, {imm});\n"
            if t == 4:
                imm = "0b" + "".join(['{0:02b}'.format(l) for l in p[::-1]])
                return f"{o} = _mm256_permute4x64_epi64({a}, {imm});\n"
            if t == 8:
                raise NotImplementedError()
            if t == 16:
                raise NotImplementedError()
            if t == 32:
                b1 = get_var_name()
                a2 = get_var_name()
                c = ", ".join([str(pp % 16) for pp in p])
                s = f"const ${self.register_name} {b1} = _mm256_set1_epi8({c});\n"
                global global_variables
                global_variables.append(s)
                return f"{a2} = _mm256_permute2x128_si256({a}, {a}, 0b0001);\n"\
                        "{o} = _mm256_shuffle_epi8({a2}, {b});"
        
        # easy case: all permutation are within the same 128bit lane
        if t == 2:
            # emit nothing
            return ""
        if t == 4:
            imm = "0b" + "".join(['{0:02b}'.format(l) for l in p[::-1]])
            return f"{o} = _mm256_permute4x64_epi64({a}, {imm});\n"
        if t == 8:
            # NOTE: cheating here. Im assumesing that the two halves are symmetric
            imm = "0b" + "".join(['{0:02b}'.format(l) for l in p[:4][::-1]])
            return f"{o} = _mm256_shuffle_epi32({a}, {imm});\n"
        if t == 16:
            raise NotImplementedError()
        if t == 32:
            b = get_var_name()
            c = ", ".join([str(pp) for pp in p])
            s = f"const {self.register_name} {b} = _mm_set1_epi8({c});\n"
            #global global_variables
            global_variables.append(s)
            return f"{o} = _mm_shuffle_epi8({a}, {b});\n"

        return ret

    def hxor(self, p: int,
             o: str,
             a: str) -> str:
        """ computes the horizonatl xor of all limbs within the given register.
        :param p: number of limbs to xor up
        :param o: output register. NOTE: is some cases this must be a variable 
                name, and not a register.
        :param a: input register 
:return TODO
        """
        if p not in [2, 4, 8, 16, 32, 64, 128, 256]:
            raise ValueError

        if p == 2:
            # return __m128i
            return f"{a} ^= _mm256_permute2x128_si256({a}, {a}, 1);\n"\
                    "{o} = _mm256_extracti128_si256({a}, 0);\n"

        if p == 256:
            # return uint1_t (technically gf2)
            return f"{a} ^= _mm256_permute2x128_si256({a}, {a}, 129);\n"\
                   f"{a} ^= _mm256_srli_si256({a}, 8);\n"\
                   f"{o} = _mm256_extract_epi64({a}, 0);\n"\
                   f"{o} = __builtin_popcountll({o}); & 1\n"\

        ret = f"{a} ^= _mm256_permute2x128_si256({a}, {a}, 1);\n"
        if p >= 4:
            # return uint64_t
            ret = f"{a} ^= _mm256_srli_si256({a}, 8);\n" + ret
            if p == 4:
                ret += f"{o} = _mm256_extract_epi64(${a}, 0);\n"
        if p >= 8:
            # return uint32_t
            ret = f"{a} ^= _mm256_srli_si256({a}, 4);\n" + ret
            if p == 8:
                ret += f"{o} = _mm256_extract_epi32({a}, 0);\n"
        if p >= 16:
            # return uint16_t
            ret = f"{a} ^= _mm256_srli_si256({a}, 2);\n" + ret
            if p == 16:
                ret += f"{o} = _mm256_extract_epi16({a}, 0);\n"
        if p >= 32:
            # return uint8_t
            ret = f"{a} ^= _mm256_srli_si256({a}, 1);\n" + ret
            if p == 32:
                ret += f"{o} = _mm256_extract_epi8({a}, 0);\n"
        if p >= 64:
            # return uint4_t (technically gf16)
            ret = f"{a} ^= _mm256_srli_epi32({a}, 4);\n" + ret
            if p == 64:
                ret += f"{o} = _mm256_extract_epi8({a}, 0) & 0xF;\n"
        if p >= 128:
            # return uint2_t (technically gf4)
            ret = f"{a} ^= _mm256_srli_epi32({a}, b);\n" + ret
            if p == 128:
                ret += f"{o} = _mm256_extract_epi8({a}, 0) & 0x3;\n"

        return ret

    @staticmethod 
    def test():
        a = AVX(16)
        #print(a.hxor(4, "o", "i"))

        #print(a.shuffle([1,0], "o", "i"))
        #print(a.shuffle([0,1], "o", "i"))
        #print(a.shuffle([1,0,3,2], "o", "i"))
        #print(a.shuffle([3,2,1,0], "o", "i"))
        assert not a.__check_lanes128([0, 1])
        assert a.__check_lanes128([1, 0])
        assert not a.__check_lanes128([1, 0, 3, 4])
        assert a.__check_lanes128([3, 4, 1, 0])
        #print(a.shuffle([1,0,3,2, 5,4,7,6], "o", "i"))
        #t = a.shuffle([1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], "o", "i")
        #print(global_variables[0])
        #print(t)
        print(a.setX("o", "i", 4))

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

    def shuffle(self, p: List[int],
                o: str,
                a: str) -> str:
        """
        :param p: a permutation of length 32/16/8/4/2
        :param o: output register 
        :param a: input register 
        :return a string which applies the permutation to the input register.
            NOTE: maybe global variables are emitted.
        """
        t = len(p)
        assert t in [2, 4, 8, 16, 32, 64]

        b = get_var_name()
        c = ", ".join([str(pp) for pp in p])
        s = f"const {self.register_name} {b} = _mm512_set1_epi8{t}({c});\n"
        global global_variables
        global_variables.append(s)

        if t == 2: # u256
            raise NotImplementedError()
        if t == 4: # u128
            raise NotImplementedError()
        return f"${o} = _mm512_permutexvar_epi{t}({b}, {a});"


    def hxor(self, p: int,
             o: str,
             a: str) -> str:
        """ computes the horizonatl xor of all limbs within the given register.
        :param p: number of limbs to xor up
        :param o: output register 
        :param a: input register 
        :return TODO
        """
        if p not in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
            raise ValueError

        if p == 2:
            # return __m256i
            return f"{o} = _mm512_castsi512_si256({a}) ^ _mm512_extracti64x4_epi64({a}, 1);\n"

        if p == 4:
            # return __m128i
            tmp = get_var_name()
            return f"__m256i {tmp} = _mm512_castsi512_si256({a}) ^ _mm512_extracti64x4_epi64({a}, 1);\n" \
                    "{o} = _mm256_extracti128_si256({tmp}, 0) ^ _mm256_extract32x4_epi32({tmp}, 1);\n"

        if p == 512:
            # return uint1_t (technically gf2)
            # TODO only available with the popcnt extension
            return f"{a} = _mm512_popcnt_epi64({a});\n" \
                   f"{o} = _mm512_reduce_add_epi64({a}) & 1;\n"

        # first shuffle low 256 with high, next shuffle low 128 with high 128
        ret = f"{a} ^= _mm512_permutexvar_epi64({a}, _mm512_setr_epi64(4,5,6,7,0,1,2,3));\n"\
              f"{a} ^= _mm512_permutex_epi64({a}, 0b01001110)"
        if p >= 8:
            # return uint64_t
            ret = f"{a} ^= _mm512_bsrli_epi128({a}, 4);\n" + ret
            if p == 8:
                ret += f"{o} = _mm256_extract_epi64(_mm512_castsi512_si256({a}), 0);\n"
        if p >= 16:
            # return uint32_t
            ret = f"{a} ^= _mm512_bsrli_epi128({a}, 2);\n" + ret
            if p == 16:
                ret += f"{o} = _mm256_extract_epi32(_mm512_castsi512_si256({a}), 0);\n"
        if p >= 32:
            # return uint16_t
            ret = f"{a} ^= _mm512_bsrli_epi128({a}, 1);\n" + ret
            if p == 32:
                ret += f"{o} = _mm256_extract_epi16(_mm512_castsi512_si256({a}), 0);\n"
        if p >= 64:
            # return uint8_t (technically gf16)
            ret = f"{a} ^= _mm512_bsrli_epi128({a}, 4);\n" + ret
            if p == 64:
                ret += f"{o} = _mm256_extract_epi8(_mm512_castsi512_si256({a}), 0) & 0xF;\n"
        if p >= 128:
            # return uint4_t (technically gf4)
            ret = f"{a} ^= _mm512_bsrli_epi128({a}, b);\n" + ret
            if p == 128:
                ret += f"{o} = _mm256_extract_epi8(_mm512_castsi512_si256({a}, 0) & 0x3;\n"
        return ret

    @staticmethod 
    def test():
        a = AVX512(16)
        print(a.shuffle([0,1], "a", "o"))
        print(a.shuffle([1,0], "a", "o"))
        print(a.shuffle([1,0,3,2], "a", "o"))


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
        :return a string with a single instruction
        """
        return f"{out_var} = vld1q_u{self.n}(({self.register_name} *)({in_var}));\n"

    def store(self,
              out_var: str,
              in_var: str) -> str:
        """
        :param out_var: variable name containing pointer to memory
        :param in_var: register variable
        :return a string with a single instruction
        """
        return f"vst1q_u(({self.register_name} *)({out_var}), {in_var});\n"
    
    def set1(self,
             out_var: str,
             in_var: str) -> str:
        """
        :param out_var: name of the output variable (register)
        :param in_var: value to set
        :return a string with a single instruction
        """
        return f"{out_var} = vdupq_n_u{self.n}({in_var});\n"
    
    def shuffle(self, p: List[int],
                o: str,
                a: str) -> str:
        """ NOTE: cheating with `__builtin_shufflevector()`
        :param p: a permutation of length 32/16/8/4/2
        :param o: output register 
        :param a: input register 
        :return a string with a single instruction
        """
        t = len(p)
        assert t in [2, 4, 8, 16, 32, 64]
        tmp = "{" + ",".join([str(a) for a in p]) + "}"
        return f"{o} = __builtin_shufflevector({a}, {a}, {tmp});"

    @staticmethod 
    def test():
        a = NEON(16)
        a.shuffle([0,1], "a", "o")
        a.shuffle([1,0], "a", "o")
        a.shuffle([1,0,3,2], "a", "o")


class Shuffler:
    """ TODO was war hier nochmal die IDEE?
    """
    def __init__(self) -> None:
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
        self.bits_limb_q = ceil_power_of_2_for_type(self.bits_q)
        self.bits_limb_q_mu = ceil_power_of_2_for_type(self.bits_q_mu)
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
        print(s.__dict__)
        in1_simd_names = [get_var_name() for _ in range(s.number_simd_q_mu)]
        in2_simd_names = [get_var_name() for _ in range(s.number_simd_q_mu)]

        ret, arguments = generate_function_declaration(fn_name, s.limb_type_str, 3)
        ret += "{\n"
        ret += self.simd.decl(in1_simd_names + in2_simd_names)
        ret += self.simd.load_multiple(in1_simd_names, arguments[1], s.limb_per_simd_q_mu)
        ret += self.simd.load_multiple(in2_simd_names, arguments[2], s.limb_per_simd_q_mu)
        ret += self.simd.add_multiple(in1_simd_names, in1_simd_names, in2_simd_names)
        ret += self.simd.store_multiple(arguments[0], in1_simd_names, s.limb_per_simd_q_mu)
        ret += "}\n"
        return ret

    def gen_extend(self, fn_name: str = "vector_extend") -> str:
        """
        :param fn_name: name of the function to emit. Needed, as this funciton
            is called by the `Matrix` interface
        :return a string contain the vector extention
        """
        s = Shape(self.n, self.q, self.mu, self.simd, self.padding)
        print(s.__dict__)
        in1_simd_names = [get_var_name() for _ in range(s.number_simd_q_mu)]

        ret, arguments = generate_function_declaration(fn_name, s.limb_type_str, 2)
        ret += "{\n"
        ret += self.simd.decl(in1_simd_names)
        ret += self.simd.expand_multiple(in1_simd_names, arguments[1], s.expand_number_limbs)
        ret += self.simd.store_multiple(arguments[0], in1_simd_names, s.limb_per_simd_q_mu)
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

        ret, arguments = generate_function_declaration(fn_name, A_s.limb_type_str, 3)
        ret += "{\n"
        ret += A_s.simd.decl(A_simd_names)
        ret += B_s.simd.decl([B_simd_name])

        for i in range(self.m):
            # load B into register
            ret += self.simd.set1(B_simd_name, f"b[{i}]")

            # load A into registers
            ret += A_s.simd.load_multiple(A_simd_names, arguments[1], A_s.limb_per_simd_q_mu)

            # mul each 
            for r in A_simd_names:
                ret += self.simd.mul(r, r, B_simd_name)

            # store into C
            ret += self.simd.store_multiple(arguments[0], A_simd_names, A_s.limb_per_simd_q_mu)

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

        ret, arguments = generate_function_declaration(fn_name, A_s.limb_type_str, 3)
        ret += "{\n"
        ret += A_s.simd.decl(A_simd_names)
        ret += B_s.simd.decl([B_simd_name])
        
        # NOTE: this is the implementation which unrolls the code over k
        for i in range(self.k):
            # TODO
            pass
        ret += "}\n"
        return ret


def test():
    SSE.test()
    AVX.test()
    AVX512.test()
    NEON.test()



if __name__ == "__main__":
    test()
    exit(0)

    #simd = AVX(16)
    #v = Vector(31, 16, 3, simd, padding=True)
    #v = Matrix(16, 4, 12, 16, 1, simd, padding=True)
    #print(v.gen_extend())

    #simd = AVX(127)
    #v = Vector(31, 127, 2, simd, padding=True)
    #print(v.gen_add())

    simd = AVX(2)
    v = Vector(33, 2, 8, simd, padding=True)
    print(v.gen_extend())

    #simd = AVX(127)
    #v = Matrix(32, 32, 32, 127, 1, simd, padding=True)
    #simd = AVX(127)
    #v = Matrix(32, 32, 32, 127, 1, simd, padding=True)
    print("""
#include <immintrin.h>
#include <stdint.h>
""")
    #print(v.gen_add())
    #print(v.gen_extend())
    #print(v.gen_matrix_vector_mul())
    #print(v.gen_matrix_matrix_mul())
