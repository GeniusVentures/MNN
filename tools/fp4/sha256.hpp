// Tools/fp4/sha256.hpp -- vendored single-header SHA-256 (public domain).
//
// Implementation follows FIPS 180-4 / RFC 6234. Derived from the
// public-domain WjCryptLib implementation by Ilya Levin and the
// well-known public-domain single-file variants; rewritten for
// C++11 (no <filesystem>, no OpenSSL) to satisfy decision D-03's
// no-new-dependency constraint for the SGFP4 injection tool.
//
// Usage:
//   std::string hex = sgfp4::sha256_hex(data, size); // 64-char lowercase
//
#ifndef TOOLS_FP4_SHA256_HPP
#define TOOLS_FP4_SHA256_HPP

#include <cstdint>
#include <cstring>
#include <string>

namespace sgfp4 {

namespace sha256_detail {

constexpr uint32_t kK[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

struct Ctx {
    uint32_t state[8];
    uint64_t bitCount;
    uint8_t  buffer[64];
    size_t   bufferLen;
};

inline void init(Ctx& ctx) {
    ctx.state[0] = 0x6a09e667u;
    ctx.state[1] = 0xbb67ae85u;
    ctx.state[2] = 0x3c6ef372u;
    ctx.state[3] = 0xa54ff53au;
    ctx.state[4] = 0x510e527fu;
    ctx.state[5] = 0x9b05688cu;
    ctx.state[6] = 0x1f83d9abu;
    ctx.state[7] = 0x5be0cd19u;
    ctx.bitCount  = 0;
    ctx.bufferLen = 0;
}

inline void transform(Ctx& ctx, const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[i * 4 + 0]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               (static_cast<uint32_t>(block[i * 4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i]        = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = ctx.state[0], b = ctx.state[1], c = ctx.state[2], d = ctx.state[3];
    uint32_t e = ctx.state[4], f = ctx.state[5], g = ctx.state[6], h = ctx.state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + ch + kK[i] + w[i];
        uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + mj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    ctx.state[0] += a;
    ctx.state[1] += b;
    ctx.state[2] += c;
    ctx.state[3] += d;
    ctx.state[4] += e;
    ctx.state[5] += f;
    ctx.state[6] += g;
    ctx.state[7] += h;
}

inline void update(Ctx& ctx, const uint8_t* data, size_t size) {
    ctx.bitCount += static_cast<uint64_t>(size) * 8u;
    while (size > 0) {
        size_t take = 64 - ctx.bufferLen;
        if (take > size) {
            take = size;
        }
        std::memcpy(ctx.buffer + ctx.bufferLen, data, take);
        ctx.bufferLen += take;
        data += take;
        size -= take;
        if (ctx.bufferLen == 64) {
            transform(ctx, ctx.buffer);
            ctx.bufferLen = 0;
        }
    }
}

inline void final(Ctx& ctx, uint8_t out[32]) {
    // The length field is the PRE-padding message bit count; freeze it and
    // stop counting the padding bytes themselves.
    const uint64_t bits = ctx.bitCount;
    ctx.bitCount        = 0;
    const uint8_t pad   = 0x80u;
    update(ctx, &pad, 1);
    const uint8_t zero = 0x00u;
    while (ctx.bufferLen != 56) {
        update(ctx, &zero, 1);
    }
    uint8_t lenBytes[8];
    for (int i = 0; i < 8; ++i) {
        lenBytes[i] = static_cast<uint8_t>((bits >> (56u - 8u * i)) & 0xFFu);
    }
    std::memcpy(ctx.buffer + 56, lenBytes, 8);
    transform(ctx, ctx.buffer);
    ctx.bufferLen = 0;
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = static_cast<uint8_t>((ctx.state[i] >> 24) & 0xFFu);
        out[i * 4 + 1] = static_cast<uint8_t>((ctx.state[i] >> 16) & 0xFFu);
        out[i * 4 + 2] = static_cast<uint8_t>((ctx.state[i] >> 8) & 0xFFu);
        out[i * 4 + 3] = static_cast<uint8_t>(ctx.state[i] & 0xFFu);
    }
}

} // namespace sha256_detail

/**
 * @brief SHA-256 digest of `size` bytes at `data`, as 64 lowercase hex chars.
 * Returns the empty string if data is null and size > 0.
 */
inline std::string sha256_hex(const uint8_t* data, size_t size) {
    if (nullptr == data && size > 0) {
        return std::string();
    }
    sha256_detail::Ctx ctx;
    sha256_detail::init(ctx);
    sha256_detail::update(ctx, data, size);
    uint8_t digest[32];
    sha256_detail::final(ctx, digest);

    static const char kHex[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(64);
    for (int i = 0; i < 32; ++i) {
        hex.push_back(kHex[(digest[i] >> 4) & 0xF]);
        hex.push_back(kHex[digest[i] & 0xF]);
    }
    return hex;
}

} // namespace sgfp4

#endif /* TOOLS_FP4_SHA256_HPP */
