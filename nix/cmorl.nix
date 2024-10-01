{ python }: with python.pkgs;

buildPythonPackage rec {
  pname = "cmorl";
  version = "0.1.0";
  src = ../.;
  doCheck = false;

  propagatedBuildInputs = [
    numpy
    pygame
    pybullet
    matplotlib
    multiprocess
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
