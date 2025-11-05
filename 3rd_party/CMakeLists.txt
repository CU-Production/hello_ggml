# stb
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE stb)

# sokol
add_library(sokol INTERFACE)
target_include_directories(sokol INTERFACE sokol)

# imgui
add_library(imgui imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp)
target_include_directories(imgui PUBLIC imgui)

# nfd
add_subdirectory(nativefiledialog-extended)

# ggml + sd.cpp
set(GGML_CCACHE OFF)
set(GGML_STATIC ON)
set(GGML_VULKAN ON)
set(SD_VULKAN ON)
add_subdirectory(stable-diffusion.cpp)

# Fix MSVC compilation issues
if(MSVC)
    # Add /bigobj flag for large object files
    target_compile_options(stable-diffusion PRIVATE /bigobj)
    # Fix UTF-8 encoding warnings
    target_compile_options(stable-diffusion PRIVATE /utf-8)
endif()