{ stdenv, python34, python34Packages, ... }:
with python34Packages;
buildPythonPackage rec {
  name = "gym";
  src = ./.;
  buildInputs = [ numpy requests six pyglet tensorflow tensorflow-tensorboard ];
}
