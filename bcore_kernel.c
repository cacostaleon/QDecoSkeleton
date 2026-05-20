__kernel void bcore_kernel(
    __global const float2* rho,
    __global       float*  D_out,
    const int level,
    const int N)
{
    int id = get_global_id(0);
    float2 r00 = rho[4*id+0], r01 = rho[4*id+1];
    float2 r10 = rho[4*id+2], r11 = rho[4*id+3];
    float purity = r00.x*r00.x + r00.y*r00.y
                 + r01.x*r01.x + r01.y*r01.y
                 + r10.x*r10.x + r10.y*r10.y
                 + r11.x*r11.x + r11.y*r11.y;
    D_out[id * N + level] = 1.0f - purity;
}