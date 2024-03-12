{ buildPythonPackage
, pythonPackages
}:


buildPythonPackage rec {
  pname = "cmorl";
  version = "0.1.0";
  src = ../.;
  doCheck = false;

  propagatedBuildInputs = with pythonPackages; [
    numpy
    pygame
    pybullet
    matplotlib
    (wandb.override {
      protobuf = null;
    })
    gymnasium
    tensorflow
    tqdm
    keras
    dm-tree
    rich
    pybox2d
    (callPackage ./mujoco-py.nix {})
  ];
}
