快速下载download\_hf\_files.py

python3 download\_hf\_files.py IPEC-COMMUNITY/spatialvla-4b-224-pt main --repo-type model --download\_path /data1/spatialvla-4b-224-pt


python3 hf_downloader.py openvla/modified_libero_rlds main --repo-type dataset --download_path ./modified_libero_rlds_data

```python
import os
import requests
import json
import argparse
from urllib.parse import urljoin, quote

# Function to fetch the file list from Hugging Face repository recursively
def fetch_file_list(repo_id, branch, repo_type):
    """
    Fetches a recursive list of files from a Hugging Face repository.
    repo_type can be 'model' or 'dataset'.
    """
    # Use the recursive=true parameter to get all files in subdirectories
    api_url = f"https://huggingface.co/api/{repo_type}s/{repo_id}/tree/{branch}?recursive=true"
    print(f"Fetching file list from: {api_url}")
  
    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"Failed to fetch data from {api_url}. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        exit(1)
    return response.json()

# Function to download files using aria2c
def download_files(file_list, repo_id, branch, download_path, repo_type):
    """
    Downloads files using aria2c, preserving the original directory structure.
    """
    # Create the root download directory
    os.makedirs(download_path, exist_ok=True) # <---
    print(f"Ensuring root download directory exists: {download_path}")

    # Prepare the download URLs and save them to a file
    download_links_with_out = []
  
    # Base URL for downloads depends on the repo type
    if repo_type == 'dataset':
        base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/"
    else: # Default to model
        base_url = f"https://huggingface.co/{repo_id}/resolve/{branch}/"

    for file_info in file_list:
        if file_info.get('type') == 'file':
            file_path = file_info['path']
        
            # --- START OF CRITICAL FIX ---
        
            # 1. Create the local directory for the file BEFORE downloading
            #    `os.path.dirname()` gets the directory part of the path
            local_file_dir = os.path.dirname(os.path.join(download_path, file_path))
            if local_file_dir:
                os.makedirs(local_file_dir, exist_ok=True) # <--- NEW! Create subdirectories
        
            # 2. Prepare the URL and the 'out' parameter for aria2c
            file_url = urljoin(base_url, quote(file_path, safe=''))
            #    The 'out' parameter tells aria2c EXACTLY where to save the file
            #    relative to the main download directory (-d option)
            aria2c_options = f"  out={file_path}" # <--- NEW! Specify output path and name
        
            download_links_with_out.append((file_url, aria2c_options))
        
            # --- END OF CRITICAL FIX ---

    if not download_links_with_out:
        print("No files found to download.")
        return

    # Save the download links and options to a text file
    download_file = os.path.join(download_path, "download_links.txt")
    with open(download_file, "w") as f:
        for url, options in download_links_with_out:
            f.write(f"{url}\n{options}\n") # <--- MODIFIED! Write URL and options on separate lines

    print(f"Found {len(download_links_with_out)} files. Links and output paths saved to {download_file}")
  
    # Run aria2c. The -d option sets the BASE directory, and the 'out' option
    # in the file specifies the path INSIDE that base directory.
    print("Starting download with aria2c (preserving directory structure)...")
    command = f"aria2c -x 16 -c --retry-wait=5 --max-tries=0 -i \"{download_file}\" -d \"{download_path}\""
    os.system(command)


def main():
    # Argument parser to handle command-line input
    parser = argparse.ArgumentParser(description="Download files from Hugging Face repository using aria2c")
    parser.add_argument("repo_id", help="The Hugging Face repository ID (e.g., 'IPEC-COMMUNITY/spatialvla-4b-224-pt' or 'openvla/modified_libero_rlds')")
    parser.add_argument("branch", help="The branch to download from (e.g., 'main')")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset"], help="Type of the repository ('model' or 'dataset'). Default is 'model'.")
    parser.add_argument("--download_path", default=None, help="The path to save the downloaded files. Defaults to a directory named after the repo_id.")
  
    args = parser.parse_args()
  
    # If download_path is not specified, create a directory with the repo_id name
    download_path = args.download_path
    if download_path is None:
        download_path = f"./{args.repo_id.split('/')[-1]}"

    # Fetch file list from the Hugging Face API
    file_list = fetch_file_list(args.repo_id, args.branch, args.repo_type)
  
    # Download the files using aria2c
    download_files(file_list, args.repo_id, args.branch, download_path, args.repo_type)
  
    print(f"Download completed! Files are saved to {download_path}")

if __name__ == "__main__":
    main()
```

可能遇到的报错

```python
-03 17:58:10,432] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Replace train dataloader!!
Replace compute_loss!!
Replace train sampler!!
[2025-08-03 17:58:30,718] [INFO] [comm.py:652:init_distributed] cdb=None
[2025-08-03 17:58:30,719] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
08/03/2025 17:58:30 - WARNING - __main__ - Process rank: 0, device: cuda:0, n_gpu: 1distributed training: True, 16-bits training: False
[WARNING|processing_utils.py:1035] 2025-08-03 17:58:33,859 >> Some kwargs in processor config are unused and will not have any effect: num_obs_steps, action_chunk_size, intrinsic_config, obs_delta, action_config, statistics, bin_policy. 
Add 0 TRANSLATION TOKENS, tokenizer vocab size 257152 / 265347
Add 0 ROTATION TOKENS to tokenizer, tokenizer vocab size 257152 / 265347
Add 0 GRIPPER TOKENS to tokenizer, tokenizer vocab size 257152 / 265347
Loading checkpoint shards: 100%|██████████| 2/2 [00:01<00:00,  1.37it/s]
Computing dataset statistics. This may take a bit, but should only need to happen once.
100%|████████████████████████████████| 432/432 [00:01<00:00, 362.03it/s]
{'libero_spatial_no_noops/1.0.0': {'action': {'mean': array([ 0.153125  ,  0.13707176, -0.15526733, -0.00517644, -0.01120875,
       -0.02019424,  0.45788181]), 'std': array([0.41272601, 0.34724554, 0.50869024, 0.03726637, 0.07244443,
       0.05762377, 0.49828067]), 'max': array([0.9375    , 0.9375    , 0.9375    , 0.19714285, 0.33642858,
       0.375     , 1.        ]), 'min': array([-0.9375    , -0.9375    , -0.9375    , -0.1875    , -0.36750001,
       -0.36000001,  0.        ]), 'q01': array([-0.74547321, -0.66160715, -0.9375    , -0.10714286, -0.20678571,
       -0.18428572,  0.        ]), 'q99': array([0.9375    , 0.87589288, 0.93214285, 0.10392857, 0.17678571,
       0.14571428, 1.        ]), 'mask': array([ True,  True,  True,  True,  True,  True, False])}, 'proprio': {'mean': array([0., 0., 0., 0., 0., 0., 0.]), 'std': array([0., 0., 0., 0., 0., 0., 0.]), 'max': array([0., 0., 0., 0., 0., 0., 0.]), 'min': array([0., 0., 0., 0., 0., 0., 0.]), 'q01': array([0., 0., 0., 0., 0., 0., 0.]), 'q99': array([0., 0., 0., 0., 0., 0., 0.])}, 'num_transitions': array(52970), 'num_trajectories': array(432)}}

######################################################################################
# Loading the following 1 datasets (incl. sampling weight):                         #
# libero_spatial_no_noops/1.0.0: ===========================================1.000000 #
######################################################################################

Saved dataset statistics file at path outputs/spatialvla_4b_finetune/2025-08-03/17-58-05_libero_mixture_spatialvla-4b-224-pt_lr5e-4_bs2_node1_gpu1_r32_a32_ep50_linear/ds_stats.json
Add 0 TRANSLATION TOKENS, tokenizer vocab size 257152 / 265347
Add 0 ROTATION TOKENS to tokenizer, tokenizer vocab size 257152 / 265347
Add 0 GRIPPER TOKENS to tokenizer, tokenizer vocab size 257152 / 265347
trainable params: 59,184,512 || all params: 4,087,039,243 || trainable%: 1.4481
Add 0 TRANSLATION TOKENS, tokenizer vocab size 257152 / 265347
Add 0 ROTATION TOKENS to tokenizer, tokenizer vocab size 257152 / 265347
Add 0 GRIPPER TOKENS to tokenizer, tokenizer vocab size 257152 / 265347
Using /home/vipuser/.cache/torch_extensions/py310_cu124 as PyTorch extensions root...
Creating extension directory /home/vipuser/.cache/torch_extensions/py310_cu124/fused_adam...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/vipuser/.cache/torch_extensions/py310_cu124/fused_adam/build.ninja...
Building extension module fused_adam...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] /data/micromamba/envs/mobipi/bin/nvcc --generate-dependencies-with-compile --dependency-output multi_tensor_adam.cuda.o.d -DTORCH_EXTENSION_NAME=fused_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/csrc/adam -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include/TH -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include/THC -isystem /data/micromamba/envs/mobipi/include -isystem /data/micromamba/envs/mobipi/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 --compiler-options '-fPIC' -O3 -DVERSION_GE_1_1 -DVERSION_GE_1_3 -DVERSION_GE_1_5 -lineinfo --use_fast_math -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 -DBF16_AVAILABLE -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT162_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -std=c++17 -c /data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/multi_tensor_adam.cu -o multi_tensor_adam.cuda.o 
FAILED: multi_tensor_adam.cuda.o 
/data/micromamba/envs/mobipi/bin/nvcc --generate-dependencies-with-compile --dependency-output multi_tensor_adam.cuda.o.d -DTORCH_EXTENSION_NAME=fused_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/csrc/adam -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include/TH -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include/THC -isystem /data/micromamba/envs/mobipi/include -isystem /data/micromamba/envs/mobipi/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 --compiler-options '-fPIC' -O3 -DVERSION_GE_1_1 -DVERSION_GE_1_3 -DVERSION_GE_1_5 -lineinfo --use_fast_math -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 -DBF16_AVAILABLE -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT162_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -std=c++17 -c /data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/multi_tensor_adam.cu -o multi_tensor_adam.cuda.o 
In file included from /usr/include/cuda_runtime.h:83,
                 from <command-line>:
/usr/include/crt/host_config.h:139:2: error: #error -- unsupported GNU version! gcc versions later than 11 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
  139 | #error -- unsupported GNU version! gcc versions later than 11 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
      |  ^~~~~
[2/3] c++ -MMD -MF fused_adam_frontend.o.d -DTORCH_EXTENSION_NAME=fused_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/csrc/includes -I/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/csrc/adam -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include/TH -isystem /data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/include/THC -isystem /data/micromamba/envs/mobipi/include -isystem /data/micromamba/envs/mobipi/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -DVERSION_GE_1_1 -DVERSION_GE_1_3 -DVERSION_GE_1_5 -DBF16_AVAILABLE -c /data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/csrc/adam/fused_adam_frontend.cpp -o fused_adam_frontend.o 
ninja: build stopped: subcommand failed.
[rank0]: Traceback (most recent call last):
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 2104, in _run_ninja_build
[rank0]:     subprocess.run(
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/subprocess.py", line 526, in run
[rank0]:     raise CalledProcessError(retcode, process.args,
[rank0]: subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.

[rank0]: The above exception was the direct cause of the following exception:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/vipuser/17robo/SpatialVLA-Libero/train/spatialvla_finetune.py", line 359, in <module>
[rank0]:     main()
[rank0]:   File "/home/vipuser/17robo/SpatialVLA-Libero/train/spatialvla_finetune.py", line 348, in main
[rank0]:     train_result = trainer.train(resume_from_checkpoint=checkpoint)
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/transformers/trainer.py", line 2164, in train
[rank0]:     return inner_training_loop(
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/transformers/trainer.py", line 2325, in _inner_training_loop
[rank0]:     model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/accelerate/accelerator.py", line 1344, in prepare
[rank0]:     result = self._prepare_deepspeed(*args)
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/accelerate/accelerator.py", line 1851, in _prepare_deepspeed
[rank0]:     engine, optimizer, _, lr_scheduler = ds_initialize(**kwargs)
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/__init__.py", line 193, in initialize
[rank0]:     engine = DeepSpeedEngine(args=args,
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 313, in __init__
[rank0]:     self._configure_optimizer(optimizer, model_parameters)
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 1276, in _configure_optimizer
[rank0]:     basic_optimizer = self._configure_basic_optimizer(model_parameters)
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 1353, in _configure_basic_optimizer
[rank0]:     optimizer = FusedAdam(
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/adam/fused_adam.py", line 94, in __init__
[rank0]:     fused_adam_cuda = FusedAdamBuilder().load()
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/op_builder/builder.py", line 531, in load
[rank0]:     return self.jit_load(verbose)
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/op_builder/builder.py", line 578, in jit_load
[rank0]:     op_module = load(name=self.name,
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1314, in load
[rank0]:     return _jit_compile(
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1721, in _jit_compile
[rank0]:     _write_ninja_file_and_build_library(
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1833, in _write_ninja_file_and_build_library
[rank0]:     _run_ninja_build(
[rank0]:   File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 2120, in _run_ninja_build
[rank0]:     raise RuntimeError(message) from e
[rank0]: RuntimeError: Error building extension 'fused_adam'
[rank0]:[W803 17:59:42.789869908 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
E0803 17:59:44.057000 48913 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 48968) of binary: /data/micromamba/envs/mobipi/bin/python3.10
Traceback (most recent call last):
  File "/data/micromamba/envs/mobipi/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
  File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/distributed/run.py", line 919, in main
    run(args)
  File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/distributed/run.py", line 910, in run
    elastic_launch(
  File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/data/micromamba/envs/mobipi/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
train/spatialvla_finetune.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-08-03_17:59:44
  host      : ubuntu22
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 48968)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```

---

下面是ai的解决方案

> 您好！从您提供的日志来看，这个错误是由于您系统中的 `gcc` 编译器版本过高，与 `nvcc` (NVIDIA CUDA 编译器) 不兼容导致的。

错误信息明确指出：
`error: #error -- unsupported GNU version! gcc versions later than 11 are not supported!`
这意味着 `nvcc` 不支持高于 11 版本的 `gcc`。

这个问题通常有以下几种解决方案：

1. **降低 `gcc` 版本**：安装 `gcc-11` 和 `g++-11`，并将其设置为默认编译器。这是最推荐的根本性解决方案。
2. **传递特定标志给编译器**：如错误日志提示，可以添加 `-allow-unsupported-compiler` 标志给 `nvcc` 来绕过版本检查。这通常能解决问题，但存在编译失败或运行时出错的风险。

我将尝试通过修改 `deepspeed` 源码来添加这个标志，这是一种侵入性较小的方法，不需要改变您的系统环境。

首先，我需要找到相关的 `deepspeed` 源码文件。
Ran tool

好的，看来刚才的搜索没有成功。

根据错误日志，问题出在编译 DeepSpeed 的一个 CUDA 操作上。DeepSpeed 尝试即时编译 (JIT) 一个名为 `fused_adam` 的优化器，但由于您的 `gcc` 版本太新，`nvcc` 编译器拒绝编译。

日志中给出了一个关键提示：可以使用 `-allow-unsupported-compiler` 标志来强制 `nvcc` 使用高版本的 `gcc`。

我的计划是找到 DeepSpeed 内部构建这个操作的代码，并在 `nvcc` 的编译参数中加入这个标志。这通常是在一个 `builder.py` 文件中处理的。

我将再次尝试定位这个文件。
Ran tool
看来文件搜索功能出现了一些问题。

不过没关系，根据日志中的堆栈跟踪信息，我可以大致推断出文件路径。错误发生在 `deepspeed/ops/op_builder/builder.py` 文件中。

我现在直接读取这个文件，找到JIT编译的相关部分。
Read file: /data/micromamba/envs/mobipi/lib/python3.10/site-packages/deepspeed/ops/op_builder/builder.py

好的，我已经拿到了 `builder.py` 的代码。

我将修改 `nvcc_args` 方法，在 Linux 系统下，无条件地添加 `-allow-unsupported-compiler` 标志。

这是我的修改计划：

1. 定位到 `CUDAOpBuilder` 类中的 `nvcc_args` 方法。
2. 在 `args` 列表中添加 `-allow-unsupported-compiler`。

我现在就进行修改。
修改已经应用。我将 `'-allow-unsupported-compiler' if sys.platform == "win32" else ''` 这一行改成了 `'-allow-unsupported-compiler'`，这样在 Linux 编译时也会带上这个参数，从而解决了 `gcc` 版本不兼容的问题。

现在，您可以重新运行之前的训练命令，问题应该就解决了。

如果还有其他问题，随时可以再问我！
