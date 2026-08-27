//
//  sgfp4_inject.cpp
//  MNN SGFP4 v2 injection tool
//
//  Created by MNN on 2026/08/26.
//  Copyright ?? 2018, Alibaba Group Holding Limited.
//
// Plan 05-02: standalone injection tool. Given a normally-converted .mnn
// plus one or more gnus-poc fp4_exporter.py --adaptive output directories,
// produce a new .mnn + merged external sidecar where each target weight
// tensor is produced by an OpType_SGFP4Dequant node. The graph-surgery
// recipe is the one proven by Plan 05-01 (test/op/SGFP4InjectTest.cpp).
//
// Plan 06-01: the entire core lives in the header-only
// tools/fp4/sgfp4_inject_core.hpp (namespace sgfp4_inject) so the Phase 6
// classic-API test can drive the injection in-process (D-12). This file is
// a thin shim.
//
// CLI: sgfp4_inject --model <path> --niche-dir <dir> [--niche-dir <dir>...]
//                   --output <path>
// Sidecar: <output>.weight
//
#include "sgfp4_inject_core.hpp"

int main(int argc, const char* argv[]) {
    return sgfp4_inject::run(argc, argv);
}
