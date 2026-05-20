__kernel void fcore_kernel(
    __global const float2* rho_in,
    __global       float2* rho_out,
    __global const float*  gamma,
    const int level)
{
    int id = get_global_id(0);
    float2 r00 = rho_in[4*id+0], r01 = rho_in[4*id+1];
    float2 r10 = rho_in[4*id+2], r11 = rho_in[4*id+3];
    float g = gamma[level];
    float sq = sqrt(1.0f - g);
    rho_out[4*id+0] = r00 + (float2)(g, 0.0f) * r11;
    rho_out[4*id+1] = (float2)(sq, 0.0f) * r01;
    rho_out[4*id+2] = (float2)(sq, 0.0f) * r10;
    rho_out[4*id+3] = (float2)(1.0f - g, 0.0f) * r11;
}