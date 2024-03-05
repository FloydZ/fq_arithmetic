with import <nixpkgs> {};
{ pkgs ? import <nixpkgs> {} }:

stdenv.mkDerivation {
  name = "fq_arithmetic";
  src = ./.;

  buildInputs = [ 
    cmake
  	git 
	clang
	gbenchmark
  ] ++ (lib.optionals pkgs.stdenv.isLinux ([
  ]));
}
