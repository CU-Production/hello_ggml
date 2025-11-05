#ifndef __SD_WRAPPER_HPP__
#define __SD_WRAPPER_HPP__

#include "stable-diffusion.h"
#include <functional>
#include <vector>
#include <string>
#include <cstdint>

// Step-by-step preview callback
// Parameters: step_number, total_steps, rgba_image_data
typedef std::function<void(int, int, const std::vector<uint8_t>&)> StepPreviewCallback;

// Wrapper class for step-by-step generation with VAE decoding
class SDStepByStepWrapper {
public:
    sd_ctx_t* ctx;
    StepPreviewCallback callback;
    
    SDStepByStepWrapper(sd_ctx_t* sd_ctx) : ctx(sd_ctx), callback(nullptr) {}
    
    void set_step_callback(StepPreviewCallback cb) {
        callback = cb;
    }
    
    // Generate with step-by-step preview
    // This uses multiple generations at different step counts
    std::vector<std::vector<uint8_t>> generate_with_steps(
        const char* prompt,
        const char* negative_prompt,
        int width, 
        int height,
        int final_steps,
        int step_interval,
        float cfg_scale,
        int64_t seed);
};

#endif // __SD_WRAPPER_HPP__

