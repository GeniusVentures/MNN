//
//  PostConverter.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <MNN/MNNDefine.h>
#include "MNN_generated.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "config.hpp"

/**
 *@brief optimize MNN net
 */
std::unique_ptr<MNN::NetT> optimizeNet(std::unique_ptr<MNN::NetT>& netT, bool forTraining, modelConfig& config, const std::vector<std::string>& expectPasses);

namespace MNN {
namespace Express {
/**
 *@brief run named PostConverter passes against originNet (definition in
 * PostConverter.cpp; declared here so standalone tests can drive passes
 * directly -- Phase 11, Plan 11-01 / consumed by Plan 11-04).
 *@return false when a pass is missing from the registry or its onExecute
 * reported failure; true otherwise (Phase 12, Plan 12-01 D-11).
 */
bool RunNetPass(const std::vector<std::string>& passes, std::unique_ptr<MNN::NetT>& originNet);
} // namespace Express
} // namespace MNN

#endif // OPTIMIZER_HPP
