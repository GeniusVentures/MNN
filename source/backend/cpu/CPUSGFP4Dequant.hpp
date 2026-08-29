//
//  CPUSGFP4Dequant.hpp
//  MNN
//
//  Created by MNN on 2026/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSGFP4Dequant_hpp
#define CPUSGFP4Dequant_hpp

#include <cstdint>
#include <vector>
#include "core/Execution.hpp"

namespace MNN {

// Decodes an SGFP4 v2 uniform-layout container (external .mnn.weight-style
// sidecar) into a float weight tensor. The container is read once at setup
// (onResize), not per-inference; onExecute decodes the held buffer via
// MNN::dequant_sgfp4_container_cpu (SGFP4DequantUtils.hpp). Additive to, and
// fully independent of, the existing E2M1 CPUFP4Dequant path.
class CPUSGFP4Dequant : public Execution {
public:
    CPUSGFP4Dequant(Backend* backend, const Op* op) : Execution(backend), mOp(op) {}
    virtual ~CPUSGFP4Dequant() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    const Op* mOp = nullptr;
    std::vector<uint8_t> mContainer;
    // Padded-crop dispatch state (Plan 09-02): dims come from
    // param->dims(); padded dims derive as ceil(dim/64)*64. Non-64-aligned
    // shapes decode through dequant_sgfp4_container_cpu_crop.
    int mPaddedDimO = 0;
    int mPaddedDimI = 0;
    bool mIsPadded = false;
};

} // namespace MNN

#endif /* CPUSGFP4Dequant_hpp */
