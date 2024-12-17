with import <nixpkgs> {};
{ pkgs ? import <nixpkgs> {} }:

stdenv.mkDerivation {
  name = "fq_arithmetic";
  src = ./.;

  buildInputs = [ 
    cmake
  	git 
	clang
	llvm # needed for symbol resolution
    gbenchmark
	jasmin-compiler
  ] ++ (lib.optionals pkgs.stdenv.isLinux ([
  ]));
}
