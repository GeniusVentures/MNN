file(GLOB_RECURSE OPTIMIZER_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(MNNConverterOpt OBJECT ${OPTIMIZER_SRC})
# MNN_compression.ph.h
add_dependencies(MNNConverterOpt MNNCompress)
target_include_directories(MNNConverterOpt PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/)
# Phase 11, Plan 11-01: InsertSGFP4Dequant.cpp includes sgfp4_encode.hpp
# (hoisted root target). Its PUBLIC include dirs normally propagate via
# target_link_libraries, but this is an OBJECT library consumed through
# $<TARGET_OBJECTS:...> -- propagate manually to be safe in all configs.
target_include_directories(MNNConverterOpt PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../../../tools/fp4
        ${CMAKE_CURRENT_LIST_DIR}/../../../../include ${CMAKE_CURRENT_LIST_DIR}/../../../../3rd_party/half)
list(APPEND MNN_CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:MNNConverterOpt>)
list(APPEND MNN_CONVERTER_BACKENDS_TARGETS MNNConverterOpt)
