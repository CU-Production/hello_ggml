# Stable Diffusion cpp

## how to use

1. download sd1.5 or flux.1 guuf models

```bash
git clone https://huggingface.co/leejet/FLUX.1-dev-gguf
# or
git clone https://huggingface.co/gpustack/stable-diffusion-v1-5-GGUF

git lfs pull
```

2. change config in program.cs

```cpp
// txt2img_cli.cpp
// Model path - update this to your stable-diffusion GGUF model path
const char* model_path = "E:/SW/ML/stable-diffusion-v1-5-gguf/stable-diffusion-v1-5-Q8_0.gguf";
// const char* model_path = "E:/SW/ML/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf";
// const char* model_path = "E:/SW/ML/FLUX.1-dev-gguf/flux1-dev-Q4_K_S.gguf";
```

3. change prompt and run

```cpp
// txt2img_cli.cpp
std::string prompt = "a photo of an astronaut riding a horse on mars";
```

