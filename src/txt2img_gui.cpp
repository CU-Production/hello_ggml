#define SOKOL_IMPL
#define SOKOL_NO_ENTRY
#define SOKOL_GLCORE
#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"
#include "imgui.h"
#include "util/sokol_imgui.h"
#include "fonts/Cousine-Regular.cpp"

#include "stable-diffusion.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <nfd.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <random>

sg_pass_action pass_action{};

// Global state for the application
struct AppState {
    // UI inputs
    char prompt[512] = "a photo of an astronaut riding a horse on mars";
    char negativePrompt[512] = "";
    int numInferenceSteps = 15;
    float guidanceScale = 7.5f;
    int nThreads = -1;  // -1 = auto-detect
    int64_t seed = 42;
    bool useRandomSeed = false;  // Use random seed for each generation
    bool enableStepPreview = false;  // Enable step-by-step preview (slower)
    int previewInterval = 5;  // Preview every N steps
    bool realTimePreview = false;  // True real-time preview (VERY slow!)
    
    // Model selection
    int modelType = 0;  // 0 = SD 1.5, 1 = FLUX
    
    // SD 1.5 Model path
    char modelPath[512] = "E:/SW/ML/stable-diffusion-v1-5-gguf/stable-diffusion-v1-5-Q8_0.gguf";
    
    // FLUX Model paths
    char fluxDiffusionModel[512] = "E:/SW/ML/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf";
    char fluxVae[512] = "E:/SW/ML/FLUX.1-dev/ae.sft";
    char fluxClipL[512] = "E:/SW/ML/flux_text_encoders/clip_l.safetensors";
    char fluxT5xxl[512] = "E:/SW/ML/flux_text_encoders/t5xxl_fp16.safetensors";
    
    // Generation state
    std::atomic<bool> isGenerating{false};
    std::atomic<bool> hasNewImage{false};
    std::string statusMessage = "Ready";
    std::string lastGenerationTime = "";
    
    // Image data
    std::vector<uint8_t> imageData;
    sg_image generatedImage = {0};
    sg_view generatedImageView = {0};
    bool imageValid = false;
    
    // Iteration history for debugging
    std::vector<std::vector<uint8_t>> iterationImages;
    int currentIterationIndex = 0;
    int totalIterations = 0;
    
    // Save state
    std::string lastPrompt = "";
    std::string lastNegativePrompt = "";
    int64_t lastUsedSeed = 0;  // Store the actual seed used in last generation
    
    // Thread
    std::thread* generationThread = nullptr;
    std::mutex dataMutex;
    
    // Stable Diffusion context (created once)
    sd_ctx_t* sd_ctx = nullptr;
    bool contextLoaded = false;
    std::string lastModelType = "";  // Track model type changes
} appState;

// Function to save the current image
void saveImage() {
    if (appState.imageData.empty()) {
        appState.statusMessage = "No image to save!";
        return;
    }
    
    // Initialize NFD
    NFD_Init();
    
    // Generate default filename
    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", now);
    
    char defaultName[256];
    snprintf(defaultName, sizeof(defaultName), "sd_image_%s_Steps%d_Scale%.1f", 
        buf, appState.numInferenceSteps, appState.guidanceScale);
    
    // Define file filters
    nfdfilteritem_t filters[2] = {
        { "PNG Image", "png" },
        { "JPEG Image", "jpg,jpeg" }
    };
    
    nfdchar_t* outPath = nullptr;
    nfdresult_t result = NFD_SaveDialog(&outPath, filters, 2, nullptr, defaultName);
    
    if (result == NFD_OKAY) {
        std::string savePath(outPath);
        NFD_FreePath(outPath);
        
        // Get file extension
        std::filesystem::path filePath(savePath);
        std::string extension = filePath.extension().string();
        
        // Convert extension to lowercase
        for (auto& c : extension) {
            c = std::tolower(c);
        }
        
        // Save based on extension
        bool saveSuccess = false;
        if (extension == ".png") {
            saveSuccess = stbi_write_png(savePath.c_str(), 512, 512, 4, 
                appState.imageData.data(), 512*4) != 0;
        } else if (extension == ".jpg" || extension == ".jpeg") {
            saveSuccess = stbi_write_jpg(savePath.c_str(), 512, 512, 4, 
                appState.imageData.data(), 100) != 0;
        } else {
            // Default to PNG if no extension or unknown extension
            savePath += ".png";
            saveSuccess = stbi_write_png(savePath.c_str(), 512, 512, 4, 
                appState.imageData.data(), 512*4) != 0;
        }
        
        if (saveSuccess) {
            char statusStr[512];
            snprintf(statusStr, sizeof(statusStr), "Image saved as %s", savePath.c_str());
            appState.statusMessage = statusStr;
        } else {
            appState.statusMessage = "Failed to save image!";
        }
    } else if (result == NFD_CANCEL) {
        appState.statusMessage = "Save cancelled";
    } else {
        char statusStr[512];
        snprintf(statusStr, sizeof(statusStr), "Error: %s", NFD_GetError());
        appState.statusMessage = statusStr;
    }
    
    NFD_Quit();
}

// Progress callback function
void progress_callback(int step, int steps, float time, void* data) {
    std::lock_guard<std::mutex> lock(appState.dataMutex);
    
    char statusStr[512];
    snprintf(statusStr, sizeof(statusStr), "Generating... Step %d/%d (%.1f%%) - %.2fs", 
        step, steps, (step * 100.0f) / steps, time);
    appState.statusMessage = statusStr;
}

// Generation function to run in background thread
void generateImage() {
    try {
        auto timeStart = std::chrono::high_resolution_clock::now();
        
        std::string prompt;
        std::string negativePrompt;
        std::string modelPath;
        int sample_steps;
        float cfg_scale;
        int n_threads;
        int64_t seed;
        bool useRandomSeed;
        bool enableStepPreview;
        int previewInterval;
        bool realTimePreview;
        
        bool useFlux;
        std::string fluxDiffusionModel, fluxVae, fluxClipL, fluxT5xxl;
        
        // Copy parameters from UI state
        {
            std::lock_guard<std::mutex> lock(appState.dataMutex);
            prompt = std::string(appState.prompt);
            negativePrompt = std::string(appState.negativePrompt);
            modelPath = std::string(appState.modelPath);
            useFlux = (appState.modelType == 1);
            fluxDiffusionModel = std::string(appState.fluxDiffusionModel);
            fluxVae = std::string(appState.fluxVae);
            fluxClipL = std::string(appState.fluxClipL);
            fluxT5xxl = std::string(appState.fluxT5xxl);
            sample_steps = appState.numInferenceSteps;
            cfg_scale = appState.guidanceScale;
            n_threads = appState.nThreads;
            seed = appState.seed;
            useRandomSeed = appState.useRandomSeed;
            enableStepPreview = appState.enableStepPreview;
            previewInterval = appState.previewInterval;
            realTimePreview = appState.realTimePreview;
        }
        
        // Generate random seed if requested
        if (useRandomSeed) {
            std::random_device rd;
            std::mt19937_64 gen(rd());
            std::uniform_int_distribution<int64_t> dis(0, INT64_MAX);
            seed = dis(gen);
        }
        
        // Check if model needs to be reloaded (first load or model type changed)
        std::string currentModelType = useFlux ? "FLUX" : "SD15";
        bool needsReload = !appState.contextLoaded || appState.lastModelType != currentModelType;
        
        // Load model if needed
        if (needsReload) {
            appState.statusMessage = useFlux ? "Loading FLUX model..." : "Loading SD 1.5 model...";
            
            // Free existing context if any
            if (appState.sd_ctx != nullptr) {
                free_sd_ctx(appState.sd_ctx);
                appState.sd_ctx = nullptr;
                appState.contextLoaded = false;
            }
            
            // Initialize context parameters
            sd_ctx_params_t ctx_params;
            sd_ctx_params_init(&ctx_params);
            
            ctx_params.n_threads = n_threads;
            ctx_params.wtype = SD_TYPE_COUNT;  // Use default from model
            
            if (useFlux) {
                // FLUX configuration (based on successful CLI setup)
                ctx_params.diffusion_model_path = fluxDiffusionModel.c_str();
                ctx_params.vae_path = fluxVae.c_str();
                ctx_params.clip_l_path = fluxClipL.c_str();
                ctx_params.t5xxl_path = fluxT5xxl.c_str();
                ctx_params.vae_decode_only = false;
                
                // Memory optimization (based on working CLI config)
                ctx_params.keep_clip_on_cpu = false;
                ctx_params.keep_vae_on_cpu = false;
                ctx_params.offload_params_to_cpu = true;
                
                // Use CPU RNG for Vulkan compatibility
                ctx_params.rng_type = STD_DEFAULT_RNG;
                ctx_params.free_params_immediately = false;
            } else {
                // SD 1.5 configuration
                ctx_params.model_path = modelPath.c_str();
                ctx_params.vae_decode_only = true;
                ctx_params.rng_type = CUDA_RNG;
                ctx_params.free_params_immediately = false;
            }
            
            // Create context
            appState.sd_ctx = new_sd_ctx(&ctx_params);
            
            if (appState.sd_ctx == nullptr) {
                appState.statusMessage = "Error: Failed to load model";
                appState.isGenerating = false;
                return;
            }
            
            appState.contextLoaded = true;
            appState.lastModelType = currentModelType;
        }
        
        appState.statusMessage = "Generating image...";
        
        // Clear previous iteration history
        {
            std::lock_guard<std::mutex> lock(appState.dataMutex);
            appState.iterationImages.clear();
            appState.totalIterations = 0;
            appState.currentIterationIndex = 0;
        }
        
        // Set progress callback
        sd_set_progress_callback(progress_callback, nullptr);
        
        // Step-by-step preview generation
        if (enableStepPreview && previewInterval > 0) {
            appState.statusMessage = "Generating step-by-step preview...";
            
            // Calculate preview steps
            std::vector<int> previewSteps;
            if (realTimePreview) {
                // Generate every single step from 1 to sample_steps
                for (int step = 1; step <= sample_steps; step++) {
                    previewSteps.push_back(step);
                }
            } else {
                // Generate at interval
                for (int step = previewInterval; step <= sample_steps; step += previewInterval) {
                    previewSteps.push_back(step);
                }
                if (previewSteps.empty() || previewSteps.back() != sample_steps) {
                    previewSteps.push_back(sample_steps);
                }
            }
            
            // Generate images at each preview step
            for (size_t i = 0; i < previewSteps.size(); i++) {
                int current_steps = previewSteps[i];
                
                // Skip very low step counts as they may be unstable
                if (current_steps < 1) {
                    continue;
                }
                
                {
                    std::lock_guard<std::mutex> lock(appState.dataMutex);
                    char statusStr[512];
                    snprintf(statusStr, sizeof(statusStr), "Generating preview %zu/%zu (Steps: %d)", 
                        i + 1, previewSteps.size(), current_steps);
                    appState.statusMessage = statusStr;
                }
                
                try {
                    // Initialize image generation parameters
                    sd_img_gen_params_t gen_params;
                    sd_img_gen_params_init(&gen_params);
                    
                    gen_params.prompt = prompt.c_str();
                    gen_params.negative_prompt = negativePrompt.c_str();
                    gen_params.clip_skip = -1;
                    gen_params.width = 512;
                    gen_params.height = 512;
                    // Use same seed to show progression of same image
                    // Note: Each generation is independent, so RNG is reset each time
                    gen_params.seed = seed;
                    gen_params.batch_count = 1;
                    
                    gen_params.sample_params.sample_steps = current_steps;
                    gen_params.sample_params.guidance.txt_cfg = cfg_scale;
                    
                    if (useFlux) {
                        gen_params.sample_params.sample_method = EULER;
                        gen_params.sample_params.scheduler = SIMPLE;
                    } else {
                        gen_params.sample_params.sample_method = sd_get_default_sample_method(appState.sd_ctx);
                        gen_params.sample_params.scheduler = DEFAULT;
                    }
                    
                    // Generate image
                    sd_image_t* results = generate_image(appState.sd_ctx, &gen_params);
                    
                    if (results == nullptr) {
                        std::lock_guard<std::mutex> lock(appState.dataMutex);
                        char errorStr[512];
                        snprintf(errorStr, sizeof(errorStr), "Failed to generate preview at step %d", current_steps);
                        appState.statusMessage = errorStr;
                        continue;
                    }
                    
                    if (results[0].data != nullptr) {
                        // Convert to RGBA
                        std::vector<uint8_t> rgbaData;
                        int width = results[0].width;
                        int height = results[0].height;
                        int channels = results[0].channel;
                        
                        if (channels == 3) {
                            rgbaData.resize(width * height * 4);
                            for (int j = 0; j < width * height; j++) {
                                rgbaData[j * 4 + 0] = results[0].data[j * 3 + 0];
                                rgbaData[j * 4 + 1] = results[0].data[j * 3 + 1];
                                rgbaData[j * 4 + 2] = results[0].data[j * 3 + 2];
                                rgbaData[j * 4 + 3] = 255;
                            }
                        } else if (channels == 4) {
                            rgbaData.assign(results[0].data, results[0].data + width * height * 4);
                        }
                        
                        // Save to iteration history
                        {
                            std::lock_guard<std::mutex> lock(appState.dataMutex);
                            appState.iterationImages.push_back(rgbaData);
                            appState.totalIterations = (int)appState.iterationImages.size();
                            appState.currentIterationIndex = appState.totalIterations - 1;
                            
                            // Update current image
                            appState.imageData = rgbaData;
                            appState.hasNewImage = true;
                        }
                        
                        free(results[0].data);
                    }
                    
                    free(results);
                    
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(appState.dataMutex);
                    char errorStr[512];
                    snprintf(errorStr, sizeof(errorStr), "Error at step %d: %s", current_steps, e.what());
                    appState.statusMessage = errorStr;
                    continue;
                }
            }
            
            auto timeEnd = std::chrono::high_resolution_clock::now();
            uint64_t milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
            
            {
                std::lock_guard<std::mutex> lock(appState.dataMutex);
                appState.lastPrompt = prompt;
                appState.lastNegativePrompt = negativePrompt;
                appState.lastUsedSeed = seed;
                
                char timeStr[64];
                snprintf(timeStr, sizeof(timeStr), "%.2fs", milliseconds / 1000.0f);
                appState.lastGenerationTime = timeStr;
                
                char statusStr[512];
                snprintf(statusStr, sizeof(statusStr), "Generated %d previews in %.2fs (use slider to review)", 
                    appState.totalIterations, milliseconds / 1000.0f);
                appState.statusMessage = statusStr;
            }
            
        } else {
            // Normal single-shot generation
            sd_img_gen_params_t gen_params;
            sd_img_gen_params_init(&gen_params);
            
            gen_params.prompt = prompt.c_str();
            gen_params.negative_prompt = negativePrompt.c_str();
            gen_params.clip_skip = -1;
            gen_params.width = 512;
            gen_params.height = 512;
            gen_params.seed = seed;
            gen_params.batch_count = 1;
            
            gen_params.sample_params.sample_steps = sample_steps;
            gen_params.sample_params.guidance.txt_cfg = cfg_scale;
            
            if (useFlux) {
                gen_params.sample_params.sample_method = EULER;
                gen_params.sample_params.scheduler = SIMPLE;
            } else {
                gen_params.sample_params.sample_method = sd_get_default_sample_method(appState.sd_ctx);
                gen_params.sample_params.scheduler = DEFAULT;
            }
            
            sd_image_t* results = generate_image(appState.sd_ctx, &gen_params);
            
            auto timeEnd = std::chrono::high_resolution_clock::now();
            uint64_t milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
            
            if (results != nullptr && results[0].data != nullptr) {
                std::vector<uint8_t> rgbaData;
                int width = results[0].width;
                int height = results[0].height;
                int channels = results[0].channel;
                
                if (channels == 3) {
                    rgbaData.resize(width * height * 4);
                    for (int i = 0; i < width * height; i++) {
                        rgbaData[i * 4 + 0] = results[0].data[i * 3 + 0];
                        rgbaData[i * 4 + 1] = results[0].data[i * 3 + 1];
                        rgbaData[i * 4 + 2] = results[0].data[i * 3 + 2];
                        rgbaData[i * 4 + 3] = 255;
                    }
                } else if (channels == 4) {
                    rgbaData.assign(results[0].data, results[0].data + width * height * 4);
                }
                
                {
                    std::lock_guard<std::mutex> lock(appState.dataMutex);
                    appState.imageData = std::move(rgbaData);
                    appState.hasNewImage = true;
                    appState.lastPrompt = prompt;
                    appState.lastNegativePrompt = negativePrompt;
                    appState.lastUsedSeed = seed;
                    
                    char timeStr[64];
                    snprintf(timeStr, sizeof(timeStr), "%.2fs", milliseconds / 1000.0f);
                    appState.lastGenerationTime = timeStr;
                    
                    char statusStr[512];
                    snprintf(statusStr, sizeof(statusStr), "Image generated in %.2fs (click Save to export)", 
                        milliseconds / 1000.0f);
                    appState.statusMessage = statusStr;
                }
                
                free(results[0].data);
            } else {
                appState.statusMessage = "Error: Failed to generate image";
            }
            
            if (results != nullptr) {
                free(results);
            }
        }
        
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(appState.dataMutex);
        appState.statusMessage = std::string("Error: ") + e.what();
    }
    
    appState.isGenerating = false;
}

void init() {
    sg_desc desc = {};
    desc.environment = sglue_environment();
    desc.logger.func = slog_func;
    sg_setup(&desc);

    simgui_desc_t simgui_desc = {};
    simgui_desc.no_default_font = true;
    simgui_setup(&simgui_desc);

    ImGui::CreateContext();
    ImGuiIO* io = &ImGui::GetIO();
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    //io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    //io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    //io->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows
    //io->ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleFonts;
    //io->ConfigFlags |= ImGuiConfigFlags_ViewportsNoTaskBarIcons;
    //io->ConfigFlags |= ImGuiConfigFlags_ViewportsNoMerge;


    // IMGUI Font texture init
    if( !ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF(Cousine_Regular_compressed_data, Cousine_Regular_compressed_size, 18.f) )
    {
        ImGui::GetIO().Fonts->AddFontDefault();
    }

    pass_action.colors[0].load_action = SG_LOADACTION_CLEAR;
    pass_action.colors[0].clear_value = {0.45f, 0.55f, 0.60f, 1.00f};
}

void frame() {
    const int width = sapp_width();
    const int height = sapp_height();

    sg_pass pass{};
    pass.action = pass_action;
    pass.swapchain = sglue_swapchain();
    sg_begin_pass(&pass);

    simgui_new_frame({ width, height, sapp_frame_duration(), sapp_dpi_scale() });

    // Check if there's a new image to upload to GPU
    if (appState.hasNewImage.load()) {
        std::lock_guard<std::mutex> lock(appState.dataMutex);
        if (!appState.imageData.empty()) {
            // Destroy old image and view if exists
            if (appState.imageValid) {
                if (appState.generatedImageView.id != 0) {
                    sg_destroy_view(appState.generatedImageView);
                    appState.generatedImageView.id = 0;
                }
                if (appState.generatedImage.id != 0) {
                    sg_destroy_image(appState.generatedImage);
                    appState.generatedImage.id = 0;
                }
            }
            
            // Create new image
            sg_image_desc img_desc{};
            img_desc.width = 512;
            img_desc.height = 512;
            img_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
            img_desc.data.mip_levels[0].ptr = appState.imageData.data();
            img_desc.data.mip_levels[0].size = appState.imageData.size();
            
            appState.generatedImage = sg_make_image(&img_desc);
            
            // Create view for the image
            sg_view_desc view_desc{};
            view_desc.texture.image = appState.generatedImage;
            appState.generatedImageView = sg_make_view(&view_desc);
            
            appState.imageValid = true;
            appState.hasNewImage = false;
        }
    }

    // Main window
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(450, height), ImGuiCond_Always);
    ImGui::Begin("Stable Diffusion Text to Image", nullptr, 
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
    
    ImGui::SeparatorText("Prompt");
    ImGui::InputTextMultiline("##prompt", appState.prompt, sizeof(appState.prompt), 
        ImVec2(-1, 60), ImGuiInputTextFlags_WordWrap);
    
    ImGui::Text("Negative Prompt:");
    ImGui::InputTextMultiline("##negative_prompt", appState.negativePrompt, sizeof(appState.negativePrompt), 
        ImVec2(-1, 40), ImGuiInputTextFlags_WordWrap);
    
    ImGui::SeparatorText("Generation Parameters");
    
    // Show recommended parameters based on model type
    if (appState.modelType == 1) {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "FLUX Recommended: 20 steps, CFG 1.0");
    } else {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "SD 1.5 Recommended: 15 steps, CFG 7.5");
    }
    
    ImGui::SliderInt("Inference Steps", &appState.numInferenceSteps, 1, 100);
    ImGui::SliderFloat("Guidance Scale", &appState.guidanceScale, 1.0f, 20.0f, "%.1f");
    
    ImGui::InputInt("Threads", &appState.nThreads);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("-1 = auto-detect CPU cores");
    }
    
    // Seed configuration
    ImGui::Checkbox("Random Seed", &appState.useRandomSeed);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Generate a new random seed for each generation");
    }
    
    if (!appState.useRandomSeed) {
        ImGui::InputScalar("Seed", ImGuiDataType_S64, &appState.seed);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Fixed seed for reproducible results");
        }
    } else {
        ImGui::BeginDisabled();
        int64_t randomSeedDisplay = -1;
        ImGui::InputScalar("Seed", ImGuiDataType_S64, &randomSeedDisplay);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Seed will be randomly generated");
        }
    }
    
    ImGui::Separator();
    ImGui::Checkbox("Enable Step Preview", &appState.enableStepPreview);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Generate images at multiple step counts to see progression\n(Much slower! ~N times slower where N is steps/interval)");
    }
    
    if (appState.enableStepPreview) {
        ImGui::Checkbox("Real-Time Mode (Every Step)", &appState.realTimePreview);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Generate image after EVERY denoising step\nVERY SLOW but shows true progression!");
        }
        
        if (!appState.realTimePreview) {
            ImGui::SliderInt("Preview Interval", &appState.previewInterval, 1, 10);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Generate preview every N steps (lower = more previews but slower)");
            }
            
            int estimatedPreviews = (appState.numInferenceSteps + appState.previewInterval - 1) / appState.previewInterval;
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Will generate ~%d previews", estimatedPreviews);
            ImGui::TextWrapped("Estimated time: %.1fx normal", (float)estimatedPreviews);
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.0f, 1.0f), "Will generate %d previews (EVERY step!)", appState.numInferenceSteps);
            ImGui::TextWrapped("Estimated time: ~%dx normal (VERY SLOW!)", appState.numInferenceSteps);
            ImGui::TextWrapped("This will decode VAE %d times!", appState.numInferenceSteps);
        }
    }
    
    ImGui::SeparatorText("Model Configuration");
    
    // Model type selection
    ImGui::Text("Model Type:");
    ImGui::RadioButton("Stable Diffusion 1.5", &appState.modelType, 0);
    ImGui::SameLine();
    ImGui::RadioButton("FLUX.1-dev", &appState.modelType, 1);
    
    ImGui::Separator();
    
    if (appState.modelType == 0) {
        // SD 1.5 Configuration
        if (ImGui::TreeNode("SD 1.5 Model Path")) {
            ImGui::InputText("Model (GGUF)", appState.modelPath, sizeof(appState.modelPath));
            ImGui::SameLine();
            if (ImGui::SmallButton("Browse##sd15")) {
                nfdchar_t* outPath = nullptr;
                nfdfilteritem_t filters[1] = {{"GGUF Models", "gguf"}};
                nfdresult_t result = NFD_OpenDialog(&outPath, filters, 1, nullptr);
                if (result == NFD_OKAY) {
                    strncpy(appState.modelPath, outPath, sizeof(appState.modelPath) - 1);
                    NFD_FreePath(outPath);
                }
            }
            ImGui::TextWrapped("Single GGUF file for Stable Diffusion 1.5");
            ImGui::TreePop();
        }
    } else {
        // FLUX Configuration
        if (ImGui::TreeNode("FLUX Model Paths (4 files required)")) {
            ImGui::Text("Diffusion Model (GGUF):");
            ImGui::InputText("##flux_diff", appState.fluxDiffusionModel, sizeof(appState.fluxDiffusionModel));
            ImGui::SameLine();
            if (ImGui::SmallButton("Browse##flux_diff")) {
                nfdchar_t* outPath = nullptr;
                nfdfilteritem_t filters[1] = {{"GGUF Models", "gguf"}};
                nfdresult_t result = NFD_OpenDialog(&outPath, filters, 1, nullptr);
                if (result == NFD_OKAY) {
                    strncpy(appState.fluxDiffusionModel, outPath, sizeof(appState.fluxDiffusionModel) - 1);
                    NFD_FreePath(outPath);
                }
            }
            
            ImGui::Text("VAE (.sft):");
            ImGui::InputText("##flux_vae", appState.fluxVae, sizeof(appState.fluxVae));
            ImGui::SameLine();
            if (ImGui::SmallButton("Browse##flux_vae")) {
                nfdchar_t* outPath = nullptr;
                nfdfilteritem_t filters[1] = {{"VAE Files", "sft"}};
                nfdresult_t result = NFD_OpenDialog(&outPath, filters, 1, nullptr);
                if (result == NFD_OKAY) {
                    strncpy(appState.fluxVae, outPath, sizeof(appState.fluxVae) - 1);
                    NFD_FreePath(outPath);
                }
            }
            
            ImGui::Text("CLIP-L (.safetensors):");
            ImGui::InputText("##flux_clip_l", appState.fluxClipL, sizeof(appState.fluxClipL));
            ImGui::SameLine();
            if (ImGui::SmallButton("Browse##flux_clip_l")) {
                nfdchar_t* outPath = nullptr;
                nfdfilteritem_t filters[1] = {{"SafeTensors", "safetensors"}};
                nfdresult_t result = NFD_OpenDialog(&outPath, filters, 1, nullptr);
                if (result == NFD_OKAY) {
                    strncpy(appState.fluxClipL, outPath, sizeof(appState.fluxClipL) - 1);
                    NFD_FreePath(outPath);
                }
            }
            
            ImGui::Text("T5-XXL (.safetensors):");
            ImGui::InputText("##flux_t5xxl", appState.fluxT5xxl, sizeof(appState.fluxT5xxl));
            ImGui::SameLine();
            if (ImGui::SmallButton("Browse##flux_t5xxl")) {
                nfdchar_t* outPath = nullptr;
                nfdfilteritem_t filters[1] = {{"SafeTensors", "safetensors"}};
                nfdresult_t result = NFD_OpenDialog(&outPath, filters, 1, nullptr);
                if (result == NFD_OKAY) {
                    strncpy(appState.fluxT5xxl, outPath, sizeof(appState.fluxT5xxl) - 1);
                    NFD_FreePath(outPath);
                }
            }
            
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "FLUX: ~35GB disk, 12GB+ VRAM, slower generation");
            ImGui::TreePop();
        }
    }
    
    ImGui::Separator();
    
    // Generate button
    bool generating = appState.isGenerating.load();
    // Quick parameter presets
    if (ImGui::Button("Quick Preset")) {
        ImGui::OpenPopup("preset_popup");
    }
    if (ImGui::BeginPopup("preset_popup")) {
        if (appState.modelType == 1) {
            ImGui::Text("FLUX Presets:");
            if (ImGui::Selectable("Fast (512x512, 15 steps)")) {
                appState.numInferenceSteps = 15;
                appState.guidanceScale = 1.0f;
            }
            if (ImGui::Selectable("Standard (512x512, 20 steps)")) {
                appState.numInferenceSteps = 20;
                appState.guidanceScale = 1.0f;
            }
            if (ImGui::Selectable("Quality (1024x1024, 25 steps)")) {
                appState.numInferenceSteps = 25;
                appState.guidanceScale = 1.5f;
            }
        } else {
            ImGui::Text("SD 1.5 Presets:");
            if (ImGui::Selectable("Fast (15 steps, CFG 7.5)")) {
                appState.numInferenceSteps = 15;
                appState.guidanceScale = 7.5f;
            }
            if (ImGui::Selectable("Standard (25 steps, CFG 7.5)")) {
                appState.numInferenceSteps = 25;
                appState.guidanceScale = 7.5f;
            }
            if (ImGui::Selectable("Quality (50 steps, CFG 9.0)")) {
                appState.numInferenceSteps = 50;
                appState.guidanceScale = 9.0f;
            }
        }
        ImGui::EndPopup();
    }
    
    if (generating) {
        ImGui::BeginDisabled();
    }
    
    std::string buttonText = (appState.modelType == 1) ? "Generate Image (FLUX)" : "Generate Image (SD 1.5)";
    if (ImGui::Button(buttonText.c_str(), ImVec2(-1, 40))) {
        // Start generation in background thread
        if (appState.generationThread != nullptr) {
            if (appState.generationThread->joinable()) {
                appState.generationThread->join();
            }
            delete appState.generationThread;
        }
        
        appState.isGenerating = true;
        appState.generationThread = new std::thread(generateImage);
    }
    
    if (generating) {
        ImGui::EndDisabled();
    }
    
    // Save button
    if (appState.imageData.empty() || generating) {
        ImGui::BeginDisabled();
    }
    
    if (ImGui::Button("Save Image", ImVec2(-1, 30))) {
        saveImage();
    }
    
    if (appState.imageData.empty() || generating) {
        ImGui::EndDisabled();
    }
    
    // Status
    ImGui::Separator();
    if (generating) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Status: %s", appState.statusMessage.c_str());
        ImGui::ProgressBar(-1.0f * ImGui::GetTime(), ImVec2(-1, 0));
    } else {
        ImGui::Text("Status: %s", appState.statusMessage.c_str());
    }
    
    if (!appState.lastGenerationTime.empty()) {
        ImGui::Text("Last generation time: %s", appState.lastGenerationTime.c_str());
        ImGui::Text("Used seed: %lld", (long long)appState.lastUsedSeed);
        ImGui::SameLine();
        if (ImGui::SmallButton("Copy##seed")) {
            ImGui::SetClipboardText(std::to_string(appState.lastUsedSeed).c_str());
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Copy seed to clipboard");
        }
    }
    
    ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("FPS: %.1f", io.Framerate);
    
    ImGui::End();
    
    // Image display window
    ImGui::SetNextWindowPos(ImVec2(450, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(width - 450, height), ImGuiCond_Always);
    ImGui::Begin("Generated Image", nullptr, 
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
    
    if (appState.imageValid && appState.generatedImageView.id != 0) {
        // Iteration slider (only show if we have iteration history)
        if (appState.totalIterations > 0) {
            ImGui::Text("Iteration Playback:");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Step %d/%d", 
                appState.currentIterationIndex + 1, appState.totalIterations);
            
            int sliderValue = appState.currentIterationIndex;
            if (ImGui::SliderInt("##iteration", &sliderValue, 0, appState.totalIterations - 1, "")) {
                // User changed the slider, update displayed image
                if (sliderValue >= 0 && sliderValue < appState.iterationImages.size()) {
                    std::lock_guard<std::mutex> lock(appState.dataMutex);
                    appState.currentIterationIndex = sliderValue;
                    appState.imageData = appState.iterationImages[sliderValue];
                    appState.hasNewImage = true;
                }
            }
            
            // Add quick navigation buttons
            ImGui::SameLine();
            if (ImGui::Button("<<")) {
                // Go to first
                if (appState.totalIterations > 0) {
                    std::lock_guard<std::mutex> lock(appState.dataMutex);
                    appState.currentIterationIndex = 0;
                    appState.imageData = appState.iterationImages[0];
                    appState.hasNewImage = true;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("<")) {
                // Go to previous
                if (appState.currentIterationIndex > 0) {
                    std::lock_guard<std::mutex> lock(appState.dataMutex);
                    appState.currentIterationIndex--;
                    appState.imageData = appState.iterationImages[appState.currentIterationIndex];
                    appState.hasNewImage = true;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button(">")) {
                // Go to next
                if (appState.currentIterationIndex < appState.totalIterations - 1) {
                    std::lock_guard<std::mutex> lock(appState.dataMutex);
                    appState.currentIterationIndex++;
                    appState.imageData = appState.iterationImages[appState.currentIterationIndex];
                    appState.hasNewImage = true;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button(">>")) {
                // Go to last
                if (appState.totalIterations > 0) {
                    std::lock_guard<std::mutex> lock(appState.dataMutex);
                    appState.currentIterationIndex = appState.totalIterations - 1;
                    appState.imageData = appState.iterationImages[appState.currentIterationIndex];
                    appState.hasNewImage = true;
                }
            }
            
            ImGui::Separator();
        }
        
        ImVec2 windowSize = ImGui::GetContentRegionAvail();
        float imageSize = std::min(windowSize.x, windowSize.y);
        
        // Center the image
        ImVec2 imagePos = ImGui::GetCursorPos();
        imagePos.x += (windowSize.x - imageSize) * 0.5f;
        imagePos.y += (windowSize.y - imageSize) * 0.5f;
        ImGui::SetCursorPos(imagePos);

        // Use the stored view to get ImTextureID
        ImTextureID imtex_id = simgui_imtextureid(appState.generatedImageView);

        ImGui::Image(imtex_id, ImVec2(imageSize, imageSize));
    } else {
        ImVec2 windowSize = ImGui::GetContentRegionAvail();
        ImGui::SetCursorPos(ImVec2(windowSize.x * 0.5f - 100, windowSize.y * 0.5f - 10));
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No image generated yet");
    }
    
    ImGui::End();

    simgui_render();

    sg_end_pass();
    sg_commit();
}

void cleanup() {
    // Wait for generation thread to finish
    if (appState.generationThread != nullptr) {
        if (appState.generationThread->joinable()) {
            appState.generationThread->join();
        }
        delete appState.generationThread;
        appState.generationThread = nullptr;
    }
    
    // Free stable diffusion context
    if (appState.sd_ctx != nullptr) {
        free_sd_ctx(appState.sd_ctx);
        appState.sd_ctx = nullptr;
    }
    
    // Clean up image and view
    if (appState.imageValid) {
        if (appState.generatedImageView.id != 0) {
            sg_destroy_view(appState.generatedImageView);
        }
        if (appState.generatedImage.id != 0) {
            sg_destroy_image(appState.generatedImage);
        }
    }
    
    simgui_shutdown();
    sg_shutdown();
}

void input(const sapp_event* event) {
    simgui_handle_event(event);
}

int main(int argc, const char* argv[]) {
    sapp_desc desc = {};
    desc.init_cb = init;
    desc.frame_cb = frame;
    desc.cleanup_cb = cleanup;
    desc.event_cb = input;
    desc.width = 1280;
    desc.height = 720;
    desc.high_dpi = true;
    desc.window_title = "Stable Diffusion Text to Image";
    desc.icon.sokol_default = true;
    desc.logger.func = slog_func;
    sapp_run(desc);

    return 0;
}
