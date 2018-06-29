{ stdenv, python3, python3Packages, ... }:
with python3Packages;
buildPythonPackage rec {
  name = "gym";
  src = ./.;
  buildInputs = [ numpy requests six pyglet ];
}
