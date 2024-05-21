self: super:
let
  depFromTensorflow = dep: # get a dependency from tensorflow
    builtins.head (builtins.filter (x: x.pname == dep) self.tensorflow.requiredPythonModules);
in
{  
  mujoco-py = (self.callPackage ./mujoco-py.nix {});
  wandb = super.wandb.override {protobuf = depFromTensorflow "protobuf";};
}