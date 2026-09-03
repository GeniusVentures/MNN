//
//  RemoveParams.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
#include <fstream>

#include "MNN/SGFP4DequantUtils.hpp"


template <typename T>
static void storeWeight(std::ofstream* fs, std::vector<T>& weight, std::vector<int64_t>& external, int64_t& offset, bool check = true) {
    if (weight.empty() && check) {
        return;
    }
    if (external.empty()) {
        external.push_back(offset);
    }
    int64_t size = weight.size() * sizeof(T);
    fs->write(reinterpret_cast<const char*>(weight.data()), size);
    weight.clear();
    std::vector<T> empty;
    weight.swap(empty);
    external.push_back(size);
    offset += size;
}

// Aligned SGFP4 container store (Plan 08-04, SGV2-23): unlike storeWeight,
// the sidecar region is padded to a 16-byte multiple (zero-filled) before
// the shared offset advances, while `external` records the TRUE size so
// the pad stays inert to readers. Matches sgfp4_inject_core.hpp:377-389
// (the injection tool's sidecar emission convention) exactly. NOTE:
// SGFP4DequantParamT::buffer is std::vector<int8_t> per the generated
// schema (flatc [byte] -> int8_t).
static void storeSGFP4Container(std::ofstream* fs, std::vector<int8_t>& buffer, std::vector<int64_t>& external,
                                int64_t& offset) {
    if (buffer.empty()) {
        return;
    }
    if (external.empty()) {
        external.push_back(offset);
    }
    const size_t trueSize = buffer.size();
    fs->write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(trueSize));
    const size_t aligned = MNN::sgfp4_align16(trueSize);
    const size_t pad     = aligned - trueSize;
    static const char kZero = '\0';
    for (size_t i = 0; i < pad; ++i) {
        fs->put(kZero);
    }
    external.push_back(static_cast<int64_t>(trueSize));
    // No dual-source ambiguity (D-06): after externalization, external
    // {offset, true-size} is the sole source.
    buffer.clear();
    std::vector<int8_t> empty;
    buffer.swap(empty);
    offset += static_cast<int64_t>(aligned);
}

void RemoveAndStoreParam(std::unique_ptr<MNN::OpT>& op, std::ofstream* fs, int64_t& offset) {
    if (!op->externalPath.empty()) {
        return;
    }
    const auto opType = op->main.type;
    switch (opType) {
        case MNN::OpParameter_Convolution2D:
        {
            if (op->inputIndexes.size() > 1) {
                break;
            }
            auto param = op->main.AsConvolution2D();
            if (param->quanParameter) {
                storeWeight<int8_t>(fs, param->quanParameter->buffer, param->external, offset, false);
                storeWeight<float>(fs, param->quanParameter->alpha, param->external, offset, false);
                storeWeight<float>(fs, param->bias, param->external, offset, false);
                storeWeight<uint32_t>(fs, param->quanParameter->index, param->external, offset, false);
            } else {
                storeWeight<float>(fs, param->weight, param->external, offset);
                storeWeight<float>(fs, param->bias, param->external, offset);
            }
            break;
        }
        case MNN::OpParameter_Scale: {
            auto param = op->main.AsScale();
            storeWeight<float>(fs, param->scaleData, param->external, offset);
            if (!param->biasData.empty()) {
                storeWeight<float>(fs, param->biasData, param->external, offset);
            }
            break;
        }
        case MNN::OpParameter_LayerNorm: {
            auto param = op->main.AsLayerNorm();
            if (!param->gamma.empty() && !param->beta.empty()) {
                storeWeight<float>(fs, param->gamma, param->external, offset);
                storeWeight<float>(fs, param->beta, param->external, offset);
            }
            break;
        }
        case MNN::OpParameter_Blob: {
            auto param = op->main.AsBlob();
            size_t totalSize = 1;
            for (auto dim : param->dims) {
                totalSize *= dim;
            }
            if (totalSize <= 1024) {
                break;
            }
            switch (param->dataType) {
                case MNN::DataType_DT_FLOAT:
                    storeWeight<float>(fs, param->float32s, param->external, offset);
                    break;
               case MNN::DataType_DT_BFLOAT16:
                    storeWeight<uint8_t>(fs, param->uint8s, param->external, offset);
                    break;
                case MNN::DataType_DT_INT32:
                    storeWeight<int>(fs, param->int32s, param->external, offset);
                    break;
                case MNN::DataType_DT_UINT8:
                    storeWeight<uint8_t>(fs, param->uint8s, param->external, offset);
                    break;
                case MNN::DataType_DT_INT8:
                    storeWeight<int8_t>(fs, param->int8s, param->external, offset);
                    break;
                default:
                    break;
            }
            break;
        }
        case MNN::OpParameter_SGFP4DequantParam: {
            auto param = op->main.AsSGFP4DequantParam();
            storeSGFP4Container(fs, param->buffer, param->external, offset);
            break;
        }
        default:
            break;
    }
}

bool saveExternalData(std::unique_ptr<MNN::NetT>& netT, const std::string& extraFileName) {
    std::ofstream extraFile(extraFileName, std::ios::binary);
    if (!extraFile.is_open()) {
        return false;
    }
    int64_t offset = 0;
    for (auto& op : netT->oplists) {
        RemoveAndStoreParam(op, &extraFile, offset);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            RemoveAndStoreParam(op, &extraFile, offset);
        }
    }
    extraFile.close();
    return true;
}

template <typename T>
static void loadExternalData(MNN::FileLoader* fl, std::vector<T>& data, int64_t size) {
    if (0 == size) {
        return;
    }
    data.resize(size / sizeof(T));
    fl->read(reinterpret_cast<char*>(data.data()), size);
}

void loadExternalParam(std::unique_ptr<MNN::OpT>& op, MNN::FileLoader* fl) {
    std::unique_ptr<MNN::FileLoader> flp;
    if (!op->externalPath.empty()) {
        flp.reset(new MNN::FileLoader(op->externalPath.c_str()));
        fl = flp.get();
    }
    const auto opType = op->type;
    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_ConvolutionDepthwise:
        {
            auto param = op->main.AsConvolution2D();
            if (param->external.size() != 3) {
                return;
            }
            fl->offset(param->external[0]);
            if (param->quanParameter) {
                loadExternalData<int8_t>(fl, param->quanParameter->buffer, param->external[1]);
                loadExternalData<float>(fl, param->quanParameter->alpha, param->external[2]);
                if (param->external.size() > 3) {
                    loadExternalData<float>(fl, param->bias, param->external[3]);
                }
                if (param->external.size() > 4) {
                    loadExternalData<uint32_t>(fl, param->quanParameter->index, param->external[4]);
                }
            } else {
                loadExternalData<float>(fl, param->weight, param->external[1]);
                loadExternalData<float>(fl, param->bias, param->external[2]);
            }
            param->external.clear();
            break;
        }
        case MNN::OpType_Scale: {
            auto param = op->main.AsScale();
            break;
        }
        case MNN::OpType_LayerNorm: {
            auto param = op->main.AsLayerNorm();
            break;
        }
        case MNN::OpType_TrainableParam:
        case MNN::OpType_Const: {
            auto param = op->main.AsBlob();
            if (param->external.size() != 2) {
                return;
            }
            size_t totalSize = 1;
            for (auto dim : param->dims) {
                totalSize *= dim;
            }
            fl->offset(param->external[0]);
            switch (param->dataType) {
                case MNN::DataType_DT_FLOAT:
                    loadExternalData<float>(fl, param->float32s, param->external[1]);
                    break;
                case MNN::DataType_DT_INT32:
                    loadExternalData<int>(fl, param->int32s, param->external[1]);
                    break;
                case MNN::DataType_DT_UINT8:
                    loadExternalData<uint8_t>(fl, param->uint8s, param->external[1]);
                    break;
                case MNN::DataType_DT_INT8:
                    loadExternalData<int8_t>(fl, param->int8s, param->external[1]);
                    break;
                default:
                    break;
            }
            param->external.clear();
            break;
        }
        case MNN::OpType_SGFP4Dequant: {
            // Q3 symmetry: _postTreatOp calls loadExternalParam BEFORE
            // RemoveAndStoreParam, so re-convert/reload paths need this
            // read-back to repopulate `buffer` from the sidecar.
            auto param = op->main.AsSGFP4DequantParam();
            if (param->external.size() != 2) {
                return;
            }
            fl->offset(param->external[0]);
            loadExternalData<int8_t>(fl, param->buffer, param->external[1]);
            param->external.clear();
            break;
        }
        default:
            break;
    }
    op->externalPath.clear();
}
