with import <nixpkgs> {};
{ pkgs ? import <nixpkgs> {} }:
let 
  myPython = python3;
  myPythonPackages = python3Packages;
  b = [
    # needed for running python
    myPython
    myPythonPackages.virtualenv
    myPythonPackages.ipython
    myPythonPackages.pip
    myPythonPackages.wheel
    # ruff
    jetbrains.pycharm-community

    # needed for running the C code
    cmake
  	git 
	clang
	llvm # needed for symbol resolution
    gbenchmark
	jasmin-compiler
  ];
  lib-path = with pkgs; lib.makeLibraryPath b;
in
stdenv.mkDerivation {
  name = "fq_arithmetic";
  src = ./.;

  buildInputs = b ++ (lib.optionals pkgs.stdenv.isLinux ([
  ]));

  shellHook = ''
      # Allow the use of wheels.
      SOURCE_DATE_EPOCH=$(date +%s)
      # Augment the dynamic linker path
      export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
      # Setup the virtual environment if it doesn't already exist.
      VENV=.venv
      if test ! -d $VENV; then
        virtualenv $VENV
      fi
      source ./$VENV/bin/activate
      export PYTHONPATH=$PYTHONPATH:`pwd`/$VENV/${myPython.sitePackages}/
  '';
}
