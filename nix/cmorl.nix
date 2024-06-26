{ buildPythonPackage
, python3Packages
}:
let depFromTensorflow = dep: # get a dependency from tensorflow
  builtins.head (builtins.filter (x: x.pname == dep) python3Packages.tensorflow.requiredPythonModules);
in
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
    # (callPackage ./wandb.nix {})
    # (wandb.override {protobuf = depFromTensorflow "protobuf";})
    (wandb.overrideAttrs (old: {doCheck = false;}))
    # wandb
    gymnasium
    tensorflow
    tqdm
    keras
    dm-tree
    rich
    pybox2d
    matplotlib
    pylint
    # (callPackage ./pybox2d.nix {})
    # (callPackage ./stable-baselines3.nix {})
    (callPackage ./mujoco-py.nix {})
  ];
}
