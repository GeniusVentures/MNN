//
//  CPUFP4Dequant.hpp
//  MNN
//
//  Created by MNN on 2026/05/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUFP4Dequant_hpp
#define CPUFP4Dequant_hpp

#include <vector>
#include "core/Execution.hpp"

namespace MNN {

class CPUFP4Dequant : public Execution {
public:
    CPUFP4Dequant(Backend* backend) : Execution(backend) {}
    virtual ~CPUFP4Dequant() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs) override;
};

} // namespace MNN

#endif /* CPUFP4Dequant_hpp */
