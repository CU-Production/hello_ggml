#include "stable-diffusion.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <string>

int main()
{
    auto timeStart = std::chrono::high_resolution_clock::now();

    // Configuration
    std::string prompt = "a photo of an astronaut riding a horse on mars";
    std::string negative_prompt = "";
    
    // Model path - update this to your stable-diffusion GGUF model path
    const char* model_path = "E:/SW/ML/stable-diffusion-v1-5-gguf/stable-diffusion-v1-5-Q8_0.gguf";
    // const char* model_path = "E:/SW/ML/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf";
    // const char* model_path = "E:/SW/ML/FLUX.1-dev-gguf/flux1-dev-Q4_K_S.gguf";

    int n_threads = -1;  // -1 means auto-detect
    int sample_steps = 15;
    float cfg_scale = 7.5f;
    int width = 512;
    int height = 512;
    int64_t seed = 42;
    
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    
    // Initialize context parameters
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    
    ctx_params.model_path = model_path;
    ctx_params.n_threads = n_threads;
    ctx_params.wtype = SD_TYPE_COUNT;  // Use default weight type from model
    ctx_params.rng_type = CUDA_RNG;
    ctx_params.vae_decode_only = true;
    
    // Create stable diffusion context
    std::cout << "Loading model..." << std::endl;
    sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
    
    if (sd_ctx == nullptr) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    std::cout << "Model loaded successfully" << std::endl;
    
    // Initialize image generation parameters
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    
    gen_params.prompt = prompt.c_str();
    gen_params.negative_prompt = negative_prompt.c_str();
    gen_params.clip_skip = -1;  // Auto
    gen_params.width = width;
    gen_params.height = height;
    gen_params.seed = seed;
    gen_params.batch_count = 1;
    
    // Configure sampling parameters
    gen_params.sample_params.sample_steps = sample_steps;
    gen_params.sample_params.guidance.txt_cfg = cfg_scale;
    gen_params.sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
    gen_params.sample_params.scheduler = DEFAULT;
    
    // Generate image
    std::cout << "Generating image..." << std::endl;
    sd_image_t* results = generate_image(sd_ctx, &gen_params);
    
    if (results == nullptr) {
        std::cerr << "Failed to generate image" << std::endl;
        free_sd_ctx(sd_ctx);
        return 1;
    }
    
    // Save results
    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);
    
    char buf[80];
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", now);
    
    std::string png_file_path = std::string("sd_image_") + buf + "_Steps" + std::to_string(sample_steps) + 
                               "_Scale" + std::to_string((int)cfg_scale) + ".png";
    std::string jpg_file_path = std::string("sd_image_") + buf + "_Steps" + std::to_string(sample_steps) + 
                               "_Scale" + std::to_string((int)cfg_scale) + ".jpg";
    
    if (results[0].data != nullptr) {
        stbi_write_png(png_file_path.c_str(), results[0].width, results[0].height, 
                      results[0].channel, results[0].data, 0);
        stbi_write_jpg(jpg_file_path.c_str(), results[0].width, results[0].height, 
                      results[0].channel, results[0].data, 100);
        
        std::cout << "Images saved: " << png_file_path << " and " << jpg_file_path << std::endl;
        
        // Free image data
        free(results[0].data);
    }
    
    free(results);
    
    // Clean up
    free_sd_ctx(sd_ctx);
    
    auto timeEnd = std::chrono::high_resolution_clock::now();
    uint64_t milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
    std::cout << "Time taken: " << milliseconds << "ms (" << (milliseconds / 1000.0) << "s)" << std::endl;
    
    return 0;
}
