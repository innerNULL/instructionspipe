{ nixpkgs ? <nixpkgs> }:

let
  # Import nixpkgs with unfree packages enabled (for NVIDIA EULA)
  pkgs = import nixpkgs { config.allowUnfree = true; };
  cuda     = pkgs.cudatoolkit;
  cudaPkgs = pkgs.cudaPackages;

in pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc                # C/C++ compiler
    cmake              # Build system
    git                # Version control
    pkg-config         # Library discovery

    cuda               # NVIDIA CUDA toolkit
    cudaPkgs.cudnn     # cuDNN neural-network primitives

    python3            # Python interpreter
    python3Packages.numpy
    python3Packages.pycuda  # Python CUDA bindings

    openmpi            # MPI for distributed computing
  ];

  shellHook = ''
    # Set CUDA environment
    export CUDA_HOME=${cuda}
    export PATH=${cuda}/bin:$PATH
    export LD_LIBRARY_PATH=${cuda}/lib:$LD_LIBRARY_PATH
    echo "🐢 Loaded CUDA ${cuda.version} from ${cuda}"
  '';
}

