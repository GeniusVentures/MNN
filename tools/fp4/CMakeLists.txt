set(MNN_SGFP4_TOOLS "")

# SGFP4 v2 C++ encoder (Phase 9, Plan 09-01): static lib consumed by
# run_test.out's parity suites. sgfp4_inject reads pre-made containers and
# never calls the encoder, so it does NOT link sgfp4_encode. The Phase 10
# dump harness below DOES link it, exercising the shipped encoder itself.
# Phase 11, Plan 11-01: when MNN_BUILD_CONVERTER is also ON, the root
# CMakeLists.txt hoists this same target above tools/converter -- guard
# here so the lib is defined exactly once across the build.
if(NOT TARGET sgfp4_encode)
    add_library(sgfp4_encode STATIC ${CMAKE_CURRENT_LIST_DIR}/sgfp4_encode.cpp)
    target_include_directories(sgfp4_encode PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/3rd_party/half)
endif()

# Phase 10 (Plan 10-02) encode-parity harness: dump-driven transducer for
# the validation driver's C++ parity sampling (D-11). Links sgfp4_encode.
add_executable(sgfp4_encode_dump.out ${CMAKE_CURRENT_LIST_DIR}/sgfp4_encode_dump.cpp)
target_link_libraries(sgfp4_encode_dump.out sgfp4_encode)
list(APPEND MNN_SGFP4_TOOLS sgfp4_encode_dump.out)

# Explicit inject sources (no glob double-compile anti-pattern).
add_executable(sgfp4_inject.out
    ${CMAKE_CURRENT_LIST_DIR}/sgfp4_inject.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sgfp4_inject_core.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sha256.hpp)
list(APPEND MNN_SGFP4_TOOLS sgfp4_inject.out)

foreach(TARGET ${MNN_SGFP4_TOOLS})
    target_link_libraries(${TARGET} ${MNN_DEPS})
    if (MSVC)
        target_compile_definitions(${TARGET} PRIVATE "_CRT_SECURE_NO_WARNINGS")
        if (NOT MNN_BUILD_SHARED_LIBS)
            foreach (DEPEND ${MNN_DEPS})
                target_link_options(${TARGET} PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
            endforeach ()
        endif()
    endif()
endforeach()
