option(MNN_VULKAN_IMAGE "Use Image as basic type" ON)
if (MNN_VULKAN_IMAGE)
    FILE(GLOB_RECURSE MNN_Vulkan_SRC ${CMAKE_CURRENT_LIST_DIR}/image/* ${CMAKE_CURRENT_LIST_DIR}/component/* ${CMAKE_CURRENT_LIST_DIR}/runtime/* ${CMAKE_CURRENT_LIST_DIR}/vulkan/*)
else()
    FILE(GLOB_RECURSE MNN_Vulkan_SRC ${CMAKE_CURRENT_LIST_DIR}/buffer/* ${CMAKE_CURRENT_LIST_DIR}/component/* ${CMAKE_CURRENT_LIST_DIR}/runtime/* ${CMAKE_CURRENT_LIST_DIR}/vulkan/*)
endif()

if(MNN_GPU_TIME_PROFILE)
    add_definitions(-DENABLE_VULKAN_TIME_PROFILE)
ENDIF()


include_directories("./")
if(MNN_USE_SYSTEM_LIB)
    find_package(Vulkan REQUIRED)
    set(MNN_VULKAN_LIBS ${Vulkan_LIBRARIES})
else()
    add_definitions(-DMNN_USE_LIB_WRAPPER)
endif()
if(APPLE AND MNN_USE_SYSTEM_LIB)
    # The Vulkan library on Apple platforms is MoltenVK, which is implemented
    # on top of the frameworks below (the same set MoltenVK.xcodeproj links)
    # and the Objective-C runtime. They must appear on the link line of every
    # executable linking MNN, otherwise MoltenVK's symbols are undefined.
    set(_MVK_FRAMEWORKS CoreFoundation CoreGraphics Foundation IOSurface Metal QuartzCore)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        list(APPEND _MVK_FRAMEWORKS UIKit)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        list(APPEND _MVK_FRAMEWORKS AppKit IOKit)
    endif()
    foreach(fw IN LISTS _MVK_FRAMEWORKS)
        list(APPEND MNN_VULKAN_LIBS "-framework ${fw}")
    endforeach()
    list(APPEND MNN_VULKAN_LIBS "-lobjc")
endif()
set(MNN_VULKAN_LIBS ${MNN_VULKAN_LIBS} PARENT_SCOPE)
IF(MNN_SEP_BUILD)
    if (MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
    endif()
    add_library(
        MNN_Vulkan
        SHARED
        ${MNN_Vulkan_SRC}
        )
    target_link_libraries(MNN_Vulkan MNN)
    if (MNN_USE_SYSTEM_LIB)
        target_link_libraries(MNN_Vulkan ${Vulkan_LIBRARY})
    else()
        target_link_libraries(MNN_Vulkan ${CMAKE_DL_LIBS})
    endif()
ELSE()
    add_library(
        MNN_Vulkan
        OBJECT
        ${MNN_Vulkan_SRC}
        )
ENDIF()

include_directories("${CMAKE_CURRENT_LIST_DIR}/include/" "${CMAKE_CURRENT_LIST_DIR}/component/" "${CMAKE_CURRENT_LIST_DIR}/runtime" "${CMAKE_CURRENT_LIST_DIR}/schema/current")
if (MNN_VULKAN_IMAGE)
    include_directories("${CMAKE_CURRENT_LIST_DIR}/image/execution" "${CMAKE_CURRENT_LIST_DIR}/image/backend" "${CMAKE_CURRENT_LIST_DIR}/image" "${CMAKE_CURRENT_LIST_DIR}/image/shaders")
else()
    include_directories("${CMAKE_CURRENT_LIST_DIR}/buffer/execution" "${CMAKE_CURRENT_LIST_DIR}/buffer/backend" "${CMAKE_CURRENT_LIST_DIR}/buffer" "${CMAKE_CURRENT_LIST_DIR}/buffer/shaders")
endif()

if (CMAKE_SYSTEM_NAME MATCHES "^Android")
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR)
endif()
