{ buildPythonPackage
, python3Packages
, callPackage
}:

buildPythonPackage rec {
  pname = "cmorl";
  version = "0.1.0";
  src = ../.;
  doCheck = false;

  propagatedBuildInputs = with python3Packages; [
    numpy
    pygame
    pybullet
    matplotlib
    wandb
    gymnasium
    mujoco-py
    tensorflow
    tqdm
    keras
    optree
    ml-dtypes
    rich
    pybox2d
    matplotlib
  ];
}
