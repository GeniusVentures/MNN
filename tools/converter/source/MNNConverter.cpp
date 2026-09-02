//
//  MNNConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cli.hpp"

int main(int argc, char *argv[]) {
    modelConfig modelPath;

    // parser command line arg
    auto res = MNN::Cli::initializeMNNConvertArgs(modelPath, argc, argv);
    if (!res) {
        // OQ1 (Phase 11): parse failure must be observable to scripts --
        // exit 1, not 0. Covers the D-05 --sgfp4 mutex, help/version paths,
        // and every other parse rejection.
        return 1;
    }
    // Convert
    bool convertOk = MNN::Cli::convertModel(modelPath);
    // Phase 12 D-11/D-12 (OQ1: strictest D-12 reading): a failed conversion
    // exits non-zero ONLY when --sgfp4 was requested, so flag-off exit codes
    // and behavior stay byte-identical to pre-change. Unconditional
    // non-zero propagation is a deliberate follow-up, intentionally not in
    // this phase.
    if (modelPath.useSGFP4 && !convertOk) {
        return 1;
    }
    return 0;
}
