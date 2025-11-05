// Patch for step-by-step preview feature
// This file contains modifications to enable real-time VAE decoding after each denoising step

#ifndef __STABLE_DIFFUSION_PATCH_HPP__
#define __STABLE_DIFFUSION_PATCH_HPP__

#include <functional>
#include <vector>
#include <cstdint>

// Step callback function type
// Parameters: step_number, total_steps, preview_image_rgba_data
typedef std::function<void(int, int, const std::vector<uint8_t>&)> sd_step_preview_cb_t;

// Global callback storage
static sd_step_preview_cb_t g_step_preview_callback = nullptr;

// Set the step preview callback
inline void sd_set_step_preview_callback(sd_step_preview_cb_t callback) {
    g_step_preview_callback = callback;
}

// Get the step preview callback
inline sd_step_preview_cb_t sd_get_step_preview_callback() {
    return g_step_preview_callback;
}

#endif // __STABLE_DIFFUSION_PATCH_HPP__

