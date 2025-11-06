#include "stable-diffusion.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <string>
#include <random>

int main()
{
    auto timeStart = std::chrono::high_resolution_clock::now();

    // Configuration
    std::string prompt = "a photo of an astronaut riding a horse on mars";
    std::string negative_prompt = "";
    
    // Choose which model to use
    bool use_flux = true;  // Set to false to use SD 1.5
    
    // SD 1.5 model path
    const char* sd15_model_path = "E:/SW/ML/stable-diffusion-v1-5-gguf/stable-diffusion-v1-5-Q8_0.gguf";
    
    // FLUX model paths
    const char* flux_diffusion_model = "E:/SW/ML/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf";
    const char* flux_vae = "E:/SW/ML/FLUX.1-dev/ae.sft";
    const char* flux_clip_l = "E:/SW/ML/flux_text_encoders/clip_l.safetensors";
    const char* flux_t5xxl = "E:/SW/ML/flux_text_encoders/t5xxl_fp16.safetensors";

    int n_threads = -1;  // -1 means auto-detect
    
    // Parameters (auto-configured based on model type)
    int sample_steps;
    float cfg_scale;
    int width;
    int height;
    
    if (use_flux) {
        // FLUX parameters - reduced to avoid Vulkan timeout
        sample_steps = 20;       // Reduced from 20 to avoid timeout
        cfg_scale = 1.0f;        // FLUX typically uses CFG scale 1.0-3.5
        width = 512;             // Reduced from 1024 to avoid timeout (can increase later)
        height = 512;            // Start with lower resolution first
        
        std::cout << "Note: Using reduced parameters to avoid Vulkan timeout." << std::endl;
        std::cout << "      Once stable, you can increase to 1024x1024 and 20-50 steps." << std::endl;
    } else {
        // SD 1.5 parameters
        sample_steps = 15;
        cfg_scale = 7.5f;
        width = 512;
        height = 512;
    }
    
    // Seed configuration
    bool use_random_seed = true;    // Set to true for random seed each run
    int64_t seed = 42;              // Fixed seed (used when use_random_seed = false)
    
    // Generate random seed if requested
    if (use_random_seed) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int64_t> dis(0, INT64_MAX);
        seed = dis(gen);
    }
    
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Model: " << (use_flux ? "FLUX.1-dev" : "Stable Diffusion 1.5") << std::endl;
    std::cout << "Seed: " << seed << (use_random_seed ? " (random)" : " (fixed)") << std::endl;
    std::cout << "Steps: " << sample_steps << ", CFG Scale: " << cfg_scale << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    
    // Initialize context parameters
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    
    ctx_params.n_threads = n_threads;
    ctx_params.wtype = SD_TYPE_COUNT;  // Use default weight type from model
    
    if (use_flux) {
        // FLUX configuration
        std::cout << "Configuring FLUX model..." << std::endl;
        ctx_params.diffusion_model_path = flux_diffusion_model;
        ctx_params.vae_path = flux_vae;
        ctx_params.clip_l_path = flux_clip_l;
        ctx_params.t5xxl_path = flux_t5xxl;
        ctx_params.vae_decode_only = false;      // FLUX needs full model
        
        // Memory optimization to avoid Vulkan timeout
        ctx_params.keep_clip_on_cpu = false;     // Keep CLIP on CPU (text encoding only, fast)
        ctx_params.keep_vae_on_cpu = false;      // VAE must stay on GPU to avoid transfer issues
        ctx_params.offload_params_to_cpu = true; // Disabled: can cause transfer deadlock
        
        // Use CPU RNG to avoid CUDA_RNG conflict with Vulkan backend
        ctx_params.rng_type = STD_DEFAULT_RNG;   // Important: Use CPU RNG for Vulkan

        // Disable optimizations that may cause VAE decode hang
        ctx_params.diffusion_flash_attn = false;
        ctx_params.vae_conv_direct = false;      // Disable direct VAE conv to avoid hang
        
        std::cout << "  Diffusion model: " << flux_diffusion_model << std::endl;
        std::cout << "  VAE: " << flux_vae << std::endl;
        std::cout << "  CLIP-L: " << flux_clip_l << std::endl;
        std::cout << "  T5-XXL: " << flux_t5xxl << std::endl;
    } else {
        // SD 1.5 configuration
        ctx_params.model_path = sd15_model_path;
        ctx_params.vae_decode_only = true;
        ctx_params.rng_type = CUDA_RNG;  // SD 1.5 can use CUDA_RNG
        // ctx_params.rng_type = STD_DEFAULT_RNG;
    }
    
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
    
    if (use_flux) {
        // FLUX sampling configuration
        gen_params.sample_params.sample_method = EULER;
        gen_params.sample_params.scheduler = SIMPLE;
    } else {
        // SD 1.5 sampling configuration
        gen_params.sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
        gen_params.sample_params.scheduler = DEFAULT;
    }
    
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
        std::cout << "Used seed: " << seed << " (use this seed to reproduce the image)" << std::endl;
        
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
