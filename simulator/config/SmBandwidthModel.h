#pragma once

// Outstanding DRAM requests per active SM, calibrated so that the read
// bandwidth an N-SM group reaches in the simulator tracks a B200 measurement:
// the best of direct loads and TMA (stages 2/4/8) at each SM count, divided by
// 8 because the B200 has 8 HBM stacks and the bullet hardware models one.
//
// Measured on --bullet-hw (HBM3e_8Gb_x128.ini, 16 channels) with
// --read-bw-test, 8..148 SMs in steps of 8: factor 8 stays within 6% of the
// measurement everywhere except 64-88 SMs, where the B200 saturates early and
// factor 12 is needed. The 8-to-12 boundaries sit midway between measured
// points. scripts/sweep_read_bw.py reproduces the comparison.
//
// Recalibrate if the DRAM model or the bullet hardware changes; the table
// means nothing on the HBM4 default configuration.
constexpr int kSmBwScalingFactorAuto = 0;

inline int sm_bw_scaling_factor_for(int sm_count) {
    return (sm_count >= 60 && sm_count <= 92) ? 12 : 8;
}
