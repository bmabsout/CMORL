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
    gymnasium
    tensorflow
    tqdm
    keras
    dm-tree
    rich
    (callPackage ./pybox2d.nix {})
    (callPackage ./mujoco-py.nix {})
  ];
}