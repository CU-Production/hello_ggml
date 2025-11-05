add_executable(txt2img_cli
    txt2img_cli.cpp
)

target_link_libraries(txt2img_cli PRIVATE 
    stable-diffusion
    stb
)

target_include_directories(txt2img_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/3rd_party/stable-diffusion.cpp
    ${CMAKE_SOURCE_DIR}/3rd_party/stb
)

# GUI application
add_executable(txt2img_gui
    txt2img_gui.cpp
    fonts/Cousine-Regular.cpp
    fonts/RobotoMono-Medium.cpp
)

target_link_libraries(txt2img_gui PRIVATE
    stable-diffusion
    sokol
    imgui
    stb
    nfd
)

target_include_directories(txt2img_gui PRIVATE
    ${CMAKE_SOURCE_DIR}/3rd_party/stable-diffusion.cpp
    ${CMAKE_SOURCE_DIR}/3rd_party/sokol
    ${CMAKE_SOURCE_DIR}/3rd_party/sokol/util
    ${CMAKE_SOURCE_DIR}/3rd_party/imgui
    ${CMAKE_SOURCE_DIR}/3rd_party/stb
)

# Platform-specific settings for GUI
if(WIN32)
    target_link_libraries(txt2img_gui PRIVATE opengl32)
    # Fix MSVC UTF-8 warnings
    if(MSVC)
        target_compile_options(txt2img_gui PRIVATE /utf-8)
        set_target_properties(txt2img_gui PROPERTIES
                LINK_FLAGS "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup"
        )
    endif()
elseif(APPLE)
    target_link_libraries(txt2img_gui PRIVATE 
        "-framework Cocoa" 
        "-framework QuartzCore" 
        "-framework OpenGL"
    )
elseif(UNIX)
    target_link_libraries(txt2img_gui PRIVATE GL X11 Xi Xrandr Xxf86vm Xcursor Xinerama)
endif()

# Fix CLI UTF-8 warnings on MSVC
if(MSVC)
    target_compile_options(txt2img_cli PRIVATE /utf-8)
endif()

