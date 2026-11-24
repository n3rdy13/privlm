cmake_minimum_required(VERSION 3.14)
project(privatelm_worker C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The app's own llama.cpp tree: phones and laptop must speak the same RPC
# protocol version, so never build this against another llama.cpp.
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../local_plugins/llama_flutter_android/android/src/main/cpp/llama.cpp")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL   OFF CACHE BOOL "" FORCE)
set(GGML_RPC          ON  CACHE BOOL "" FORCE)
# Metal is on by default on macOS. For an NVIDIA GPU pass -DGGML_CUDA=ON, for
# other GPUs on Linux/Windows -DGGML_VULKAN=ON (privatelm_worker.py --gpu).
add_subdirectory("${LLAMA_DIR}/ggml" ggml)

add_executable(privatelm-worker worker.cpp)
target_link_libraries(privatelm-worker PRIVATE ggml)
if (WIN32)
    target_link_libraries(privatelm-worker PRIVATE ws2_32)
endif()
