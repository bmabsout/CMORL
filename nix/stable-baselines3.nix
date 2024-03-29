{ lib
, fetchFromGitHub
, buildPythonPackage
, autoPatchelfHook
, python
, stdenv
, gymnasium
, numpy
, torch
, cloudpickle
, pandas
, matplotlib
, pytest
}:

buildPythonPackage rec {
  pname = "stable-baselines3";
  version = "2.2.1";
#   format = "wheel";

  src = fetchFromGitHub {
    owner = "DLR-RM";
    repo = "stable-baselines3";
    rev = "v${version}";
    sha256 = "sha256-Plfjrd1lpDmuSNYXJyI/sjxVssHCzyMZSmrWWiDvroQ=";
  };

  propagatedBuildInputs = [
    numpy
    gymnasium
    torch
    cloudpickle
    pandas
    matplotlib
    pytest
  ];
  doCheck=true;

  buildInputs = [
    stdenv.cc.cc.lib
  ];

  nativeBuildInputs = [
    autoPatchelfHook
  ];

  meta = with lib; {
    homepage = "https://github.com/DLR-RM/stable-baselines3";
    license = licenses.mit;
    platforms = with platforms; (linux);
  };
}