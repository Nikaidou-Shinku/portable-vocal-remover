# Portable Vocal Remover

A Rust port of [UVR](https://github.com/Anjok07/ultimatevocalremovergui).

```console
A Rust port of UVR

Usage: pvr [OPTIONS] --input-path <INPUT>

Options:
  -i, --input-path <INPUT>    Input audio file path
  -o, --output-path <OUTPUT>  Directory to save output audio [default: .]
  -p, --preset <PRESET>       The model used, leave blank to see all available models
  -d, --directml-backend      Use DirectML backend for inference
  -c, --cuda-backend          Use CUDA backend for inference
  -t, --tensorrt-backend      Use TensorRT backend for inference
  -f, --format <FORMAT>       File format used to save results (wav/flac) [default: flac]
  -h, --help                  Print help
  -V, --version               Print version
```

Supported audio formats: WAV, FLAC, MP3.

## Build

### Build PVR CLI

```shell
git clone --recursive https://github.com/Nikaidou-Shinku/portable-vocal-remover.git
cd portable-vocal-remover
cargo build -r -p pvr
```

### Build ONNX Runtime

Checkout the source tree:

```shell
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
```

Build on Windows:

```shell
.\build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --use_cuda --cudnn_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3" --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3" --use_tensorrt --tensorrt_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3" --use_dml
```

Build on Linux:

```shell
./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --use_cuda --cudnn_home "/usr" --cuda_home "/opt/cuda" --use_tensorrt --tensorrt_home "/usr" --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
```

## Credits

- [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) - Provided great high quality models.
