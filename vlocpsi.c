#include <stdint.h>

#define MAX_N1   32
#define MAX_N2   32
#define MAX_N3   32
#define MAX_GRID (MAX_N1*MAX_N2*MAX_N3)

typedef struct { double d[8]; } v8d;
typedef struct { float  real, imag; } ComplexF;


static void fft1d_f(ComplexF *data0, int N0, int inv0) {
#pragma HLS INLINE off

    float xr0[32], xi0[32];
    float xr1[32], xi1[32];
#pragma HLS ARRAY_PARTITION variable=xr0 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=xi0 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=xr1 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=xi1 cyclic factor=16

load0:
    for (int i_load0 = 0; i_load0 < 32; i_load0++) {
#pragma HLS PIPELINE II=1
        xr0[i_load0] = data0[i_load0].real;
        xi0[i_load0] = data0[i_load0].imag;
    }

    if (N0 != 32) {
passthrough0:
        for (int i_pt0 = 0; i_pt0 < N0; i_pt0++) {
#pragma HLS PIPELINE II=1
            data0[i_pt0].real = xr0[i_pt0];
            data0[i_pt0].imag = xi0[i_pt0];
        }
        return;
    }

#pragma HLS INLINE region
    {
        float t;

        t = xr0[1];  xr0[1]  = xr0[16]; xr0[16] = t;
        t = xi0[1];  xi0[1]  = xi0[16]; xi0[16] = t;

        t = xr0[2];  xr0[2]  = xr0[8];  xr0[8]  = t;
        t = xi0[2];  xi0[2]  = xi0[8];  xi0[8]  = t;

        t = xr0[3];  xr0[3]  = xr0[24]; xr0[24] = t;
        t = xi0[3];  xi0[3]  = xi0[24]; xi0[24] = t;

        t = xr0[5];  xr0[5]  = xr0[20]; xr0[20] = t;
        t = xi0[5];  xi0[5]  = xi0[20]; xi0[20] = t;

        t = xr0[6];  xr0[6]  = xr0[12]; xr0[12] = t;
        t = xi0[6];  xi0[6]  = xi0[12]; xi0[12] = t;

        t = xr0[7];  xr0[7]  = xr0[28]; xr0[28] = t;
        t = xi0[7];  xi0[7]  = xi0[28]; xi0[28] = t;

        t = xr0[9];  xr0[9]  = xr0[18]; xr0[18] = t;
        t = xi0[9];  xi0[9]  = xi0[18]; xi0[18] = t;

        t = xr0[11]; xr0[11] = xr0[26]; xr0[26] = t;
        t = xi0[11]; xi0[11] = xi0[26]; xi0[26] = t;

        t = xr0[13]; xr0[13] = xr0[22]; xr0[22] = t;
        t = xi0[13]; xi0[13] = xi0[22]; xi0[22] = t;

        t = xr0[15]; xr0[15] = xr0[30]; xr0[30] = t;
        t = xi0[15]; xi0[15] = xi0[30]; xi0[30] = t;

        t = xr0[19]; xr0[19] = xr0[25]; xr0[25] = t;
        t = xi0[19]; xi0[19] = xi0[25]; xi0[25] = t;

        t = xr0[23]; xr0[23] = xr0[29]; xr0[29] = t;
        t = xi0[23]; xi0[23] = xi0[29]; xi0[29] = t;
    }

    static const float WR8[4]  = {1.0f, 0.70710678118f, 0.0f, -0.70710678118f};
    static const float WI8[4]  = {0.0f, 0.70710678118f, 1.0f,  0.70710678118f};

    static const float WR16[8] = {
        1.0f, 0.92387953251f, 0.70710678118f, 0.38268343236f,
        0.0f,-0.38268343236f,-0.70710678118f,-0.92387953251f
    };
    static const float WI16[8] = {
        0.0f, 0.38268343236f, 0.70710678118f, 0.92387953251f,
        1.0f, 0.92387953251f, 0.70710678118f, 0.38268343236f
    };

    static const float WR32[16]= {
        1.0f,0.98078528040f,0.92387953251f,0.83146961230f,
        0.70710678118f,0.55557023301f,0.38268343236f,0.19509032201f,
        0.0f,-0.19509032201f,-0.38268343236f,-0.55557023301f,
        -0.70710678118f,-0.83146961230f,-0.92387953251f,-0.98078528040f
    };
    static const float WI32[16]= {
        0.0f,0.19509032201f,0.38268343236f,0.55557023301f,
        0.70710678118f,0.83146961230f,0.92387953251f,0.98078528040f,
        1.0f,0.98078528040f,0.92387953251f,0.83146961230f,
        0.70710678118f,0.55557023301f,0.38268343236f,0.19509032201f
    };

    const int sf0 = inv0 ? 1 : -1;

radix4_0:
    for (int i_rad0 = 0; i_rad0 < 32; i_rad0 += 4) {
#pragma HLS PIPELINE II=1
        float ar=xr0[i_rad0],   ai=xi0[i_rad0];
        float br=xr0[i_rad0+1], bi=xi0[i_rad0+1];
        float cr=xr0[i_rad0+2], ci=xi0[i_rad0+2];
        float dr=xr0[i_rad0+3], di=xi0[i_rad0+3];

        float s0r=ar+br, s0i=ai+bi;
        float s1r=ar-br, s1i=ai-bi;
        float s2r=cr+dr, s2i=ci+di;
        float s3r=cr-dr, s3i=ci-di;

        float wr = -sf0 * s3i;
        float wi =  sf0 * s3r;

        xr1[i_rad0]   = s0r + s2r;  xi1[i_rad0]   = s0i + s2i;
        xr1[i_rad0+2] = s0r - s2r;  xi1[i_rad0+2] = s0i - s2i;
        xr1[i_rad0+1] = s1r + wr;   xi1[i_rad0+1] = s1i + wi;
        xr1[i_rad0+3] = s1r - wr;   xi1[i_rad0+3] = s1i - wi;
    }

stage8_0:
    for (int blk8_0 = 0; blk8_0 < 32; blk8_0 += 8) {
        for (int j8_0 = 0; j8_0 < 4; j8_0++) {
#pragma HLS UNROLL
            int k1 = blk8_0 + j8_0;
            int k2 = k1 + 4;

            float wr = WR8[j8_0];
            float wi = inv0 ? +WI8[j8_0] : -WI8[j8_0];

            float ur = xr1[k1], ui = xi1[k1];
            float ar = xr1[k2], ai = xi1[k2];

            float tr = wr*ar - wi*ai;
            float ti = wr*ai + wi*ar;

            xr0[k1] = ur + tr; xi0[k1] = ui + ti;
            xr0[k2] = ur - tr; xi0[k2] = ui - ti;
        }
    }

stage16_0:
    for (int blk16_0 = 0; blk16_0 < 32; blk16_0 += 16) {
        for (int j16_0 = 0; j16_0 < 8; j16_0++) {
#pragma HLS UNROLL
            int k1 = blk16_0 + j16_0;
            int k2 = k1 + 8;

            float wr = WR16[j16_0];
            float wi = inv0 ? +WI16[j16_0] : -WI16[j16_0];

            float ur = xr0[k1], ui = xi0[k1];
            float ar = xr0[k2], ai = xi0[k2];

            float tr = wr*ar - wi*ai;
            float ti = wr*ai + wi*ar;

            xr1[k1] = ur + tr; xi1[k1] = ui + ti;
            xr1[k2] = ur - tr; xi1[k2] = ui - ti;
        }
    }

stage32_0:
    for (int j32_0 = 0; j32_0 < 16; j32_0++) {
#pragma HLS UNROLL
        int k1 = j32_0;
        int k2 = j32_0 + 16;

        float wr = WR32[j32_0];
        float wi = inv0 ? +WI32[j32_0] : -WI32[j32_0];

        float ur = xr1[k1], ui = xi1[k1];
        float ar = xr1[k2], ai = xi1[k2];

        float tr = wr*ar - wi*ai;
        float ti = wr*ai + wi*ar;

        xr0[k1] = ur + tr; xi0[k1] = ui + ti;
        xr0[k2] = ur - tr; xi0[k2] = ui - ti;
    }

    const float scale = inv0 ? (1.0f / 32.0f) : 1.0f;

store0:
    for (int i_store0 = 0; i_store0 < 32; i_store0++) {
#pragma HLS PIPELINE II=1
        data0[i_store0].real = xr0[i_store0] * scale;
        data0[i_store0].imag = xi0[i_store0] * scale;
    }
}

static void z_stage_3d_core(
    const double in_re0 [MAX_N3][MAX_N2][MAX_N1],
    const double in_im0 [MAX_N3][MAX_N2][MAX_N1],
    double       out_re0[MAX_N3][MAX_N2][MAX_N1],
    double       out_im0[MAX_N3][MAX_N2][MAX_N1],
    int nr1_0, int nr2_0, int nr3_0, int inv_0,
    ComplexF lb_s[32], float tmp_re_s[32], float tmp_im_s[32])
{
#pragma HLS INLINE off

z_j0:
    for (int j0 = 0; j0 < nr2_0; j0++) {
z_i0:
        for (int i0 = 0; i0 < nr1_0; i0++) {
z_load0:
            for (int k0 = 0; k0 < nr3_0; k0++) {
#pragma HLS UNROLL factor=2
                lb_s[k0].real = (float)in_re0[k0][j0][i0];
                lb_s[k0].imag = (float)in_im0[k0][j0][i0];
            }

            fft1d_f(lb_s, nr3_0, inv_0);

z_tmp0:
            for (int k1 = 0; k1 < nr3_0; k1++) {
#pragma HLS UNROLL factor=8
                tmp_re_s[k1] = lb_s[k1].real;
                tmp_im_s[k1] = lb_s[k1].imag;
            }

z_store0:
            for (int k2 = 0; k2 < nr3_0; k2++) {
#pragma HLS PIPELINE II=1
                out_re0[k2][j0][i0] = (double)tmp_re_s[k2];
                out_im0[k2][j0][i0] = (double)tmp_im_s[k2];
            }
        }
    }
}

static void y_stage_3d_core(
    const double in_re1 [MAX_N3][MAX_N2][MAX_N1],
    const double in_im1 [MAX_N3][MAX_N2][MAX_N1],
    double       out_re1[MAX_N3][MAX_N2][MAX_N1],
    double       out_im1[MAX_N3][MAX_N2][MAX_N1],
    int nr1_1, int nr2_1, int nr3_1, int inv_1,
    ComplexF lb_s[32], float tmp_re_s[32], float tmp_im_s[32])
{
#pragma HLS INLINE off

y_k0:
    for (int k3 = 0; k3 < nr3_1; k3++) {
y_i0:
        for (int i1 = 0; i1 < nr1_1; i1++) {
y_load0:
            for (int j1 = 0; j1 < nr2_1; j1++) {
#pragma HLS UNROLL factor=2
                lb_s[j1].real = (float)in_re1[k3][j1][i1];
                lb_s[j1].imag = (float)in_im1[k3][j1][i1];
            }

            fft1d_f(lb_s, nr2_1, inv_1);

y_tmp0:
            for (int j2 = 0; j2 < nr2_1; j2++) {
#pragma HLS UNROLL factor=8
                tmp_re_s[j2] = lb_s[j2].real;
                tmp_im_s[j2] = lb_s[j2].imag;
            }

y_store0:
            for (int j3 = 0; j3 < nr2_1; j3++) {
#pragma HLS PIPELINE II=1
                out_re1[k3][j3][i1] = (double)tmp_re_s[j3];
                out_im1[k3][j3][i1] = (double)tmp_im_s[j3];
            }
        }
    }
}

static void x_fused_stage_3d(
    const double in_re2 [MAX_N3][MAX_N2][MAX_N1],
    const double in_im2 [MAX_N3][MAX_N2][MAX_N1],
    double       out_re2[MAX_N3][MAX_N2][MAX_N1],
    double       out_im2[MAX_N3][MAX_N2][MAX_N1],
    const double v_local2[MAX_N3][MAX_N2][MAX_N1],
    int nr1_2, int nr2_2, int nr3_2,
    ComplexF lb_s[32], float tmp_re_s[32], float tmp_im_s[32])
{
#pragma HLS INLINE off

    float mul_re0[32];
    float mul_im0[32];
#pragma HLS ARRAY_PARTITION variable=mul_re0 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=mul_im0 cyclic factor=8

x_k0:
    for (int k4 = 0; k4 < nr3_2; k4++) {
x_j0:
        for (int j4 = 0; j4 < nr2_2; j4++) {

x_load0:
            for (int i2 = 0; i2 < nr1_2; i2++) {
#pragma HLS UNROLL factor=2
                lb_s[i2].real = (float)in_re2[k4][j4][i2];
                lb_s[i2].imag = (float)in_im2[k4][j4][i2];
            }

            fft1d_f(lb_s, nr1_2, 1);

x_copy0:
            for (int i_copy0 = 0; i_copy0 < nr1_2; i_copy0++) {
#pragma HLS UNROLL factor=8
                mul_re0[i_copy0] = lb_s[i_copy0].real;
                mul_im0[i_copy0] = lb_s[i_copy0].imag;
            }

x_mul0:
            for (int i3 = 0; i3 < nr1_2; i3++) {
#pragma HLS PIPELINE II=1
                float vv0 = (float)v_local2[k4][j4][i3];
                mul_re0[i3] *= vv0;
                mul_im0[i3] *= vv0;
            }

x_copy1:
            for (int i_copy1 = 0; i_copy1 < nr1_2; i_copy1++) {
#pragma HLS UNROLL factor=8
                lb_s[i_copy1].real = mul_re0[i_copy1];
                lb_s[i_copy1].imag = mul_im0[i_copy1];
            }

            fft1d_f(lb_s, nr1_2, 0);

x_tmp0:
            for (int i4 = 0; i4 < nr1_2; i4++) {
#pragma HLS UNROLL factor=8
                tmp_re_s[i4] = lb_s[i4].real;
                tmp_im_s[i4] = lb_s[i4].imag;
            }

x_store0:
            for (int i5 = 0; i5 < nr1_2; i5++) {
#pragma HLS PIPELINE II=1
                out_re2[k4][j4][i5] = (double)tmp_re_s[i5];
                out_im2[k4][j4][i5] = (double)tmp_im_s[i5];
            }
        }
    }
}

static void z_stage_3d_fwd(
    const double in_re3 [MAX_N3][MAX_N2][MAX_N1],
    const double in_im3 [MAX_N3][MAX_N2][MAX_N1],
    double       out_re3[MAX_N3][MAX_N2][MAX_N1],
    double       out_im3[MAX_N3][MAX_N2][MAX_N1],
    int nr1_3, int nr2_3, int nr3_3,
    ComplexF lb_s[32], float tmp_re_s[32], float tmp_im_s[32])
{
#pragma HLS INLINE off
    z_stage_3d_core(in_re3, in_im3, out_re3, out_im3, nr1_3, nr2_3, nr3_3, 1,
                    lb_s, tmp_re_s, tmp_im_s);
}

static void z_stage_3d_inv(
    const double in_re4 [MAX_N3][MAX_N2][MAX_N1],
    const double in_im4 [MAX_N3][MAX_N2][MAX_N1],
    double       out_re4[MAX_N3][MAX_N2][MAX_N1],
    double       out_im4[MAX_N3][MAX_N2][MAX_N1],
    int nr1_4, int nr2_4, int nr3_4,
    ComplexF lb_s[32], float tmp_re_s[32], float tmp_im_s[32])
{
#pragma HLS INLINE off
    z_stage_3d_core(in_re4, in_im4, out_re4, out_im4, nr1_4, nr2_4, nr3_4, 0,
                    lb_s, tmp_re_s, tmp_im_s);
}

static void y_stage_3d_fwd(
    const double in_re5 [MAX_N3][MAX_N2][MAX_N1],
    const double in_im5 [MAX_N3][MAX_N2][MAX_N1],
    double       out_re5[MAX_N3][MAX_N2][MAX_N1],
    double       out_im5[MAX_N3][MAX_N2][MAX_N1],
    int nr1_5, int nr2_5, int nr3_5,
    ComplexF lb_s[32], float tmp_re_s[32], float tmp_im_s[32])
{
#pragma HLS INLINE off
    y_stage_3d_core(in_re5, in_im5, out_re5, out_im5, nr1_5, nr2_5, nr3_5, 1,
                    lb_s, tmp_re_s, tmp_im_s);
}

static void y_stage_3d_inv(
    const double in_re6 [MAX_N3][MAX_N2][MAX_N1],
    const double in_im6 [MAX_N3][MAX_N2][MAX_N1],
    double       out_re6[MAX_N3][MAX_N2][MAX_N1],
    double       out_im6[MAX_N3][MAX_N2][MAX_N1],
    int nr1_6, int nr2_6, int nr3_6,
    ComplexF lb_s[32], float tmp_re_s[32], float tmp_im_s[32])
{
#pragma HLS INLINE off
    y_stage_3d_core(in_re6, in_im6, out_re6, out_im6, nr1_6, nr2_6, nr3_6, 0,
                    lb_s, tmp_re_s, tmp_im_s);
}


void vloc_psi(const v8d *psi_in,
              const v8d *v,
              v8d *hpsi_out,
              const int *igk,
              int lda, int n, int m,
              int nr1, int nr2, int nr3)
{
#pragma HLS INTERFACE m_axi port=psi_in   bundle=gmem0 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=v        bundle=gmem1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=hpsi_out bundle=gmem2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=igk      bundle=gmem3

#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=lda  bundle=control
#pragma HLS INTERFACE s_axilite port=n    bundle=control
#pragma HLS INTERFACE s_axilite port=m    bundle=control
#pragma HLS INTERFACE s_axilite port=nr1  bundle=control
#pragma HLS INTERFACE s_axilite port=nr2  bundle=control
#pragma HLS INTERFACE s_axilite port=nr3  bundle=control

    ComplexF lb_s0[32];
    float tmp_re_s0[32];
    float tmp_im_s0[32];
#pragma HLS ARRAY_PARTITION variable=lb_s0 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=tmp_re_s0 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=tmp_im_s0 cyclic factor=8

    static double psi_r_re[MAX_N3][MAX_N2][MAX_N1];
    static double psi_r_im[MAX_N3][MAX_N2][MAX_N1];
    static double bufA_re [MAX_N3][MAX_N2][MAX_N1];
    static double bufA_im [MAX_N3][MAX_N2][MAX_N1];
    static double bufB_re [MAX_N3][MAX_N2][MAX_N1];
    static double bufB_im [MAX_N3][MAX_N2][MAX_N1];
    static double v_local [MAX_N3][MAX_N2][MAX_N1];

    int Ngrid0 = nr1 * nr2 * nr3;

load_v0:
    for (int idxv0 = 0; idxv0 < Ngrid0; idxv0++) {
#pragma HLS PIPELINE II=2
        int kL0 = idxv0 / (nr1 * nr2);
        int rL0 = idxv0 - kL0 * (nr1 * nr2);
        int jL0 = rL0 / nr1;
        int iL0 = rL0 - jL0 * nr1;

        v8d tmp0 = v[idxv0 >> 3];
        v_local[kL0][jL0][iL0] = tmp0.d[idxv0 & 7];
    }

state_loop0:
    for (int s0 = 0; s0 < m; ++s0) {
        int off0 = s0 * lda;

scatter0:
        for (int g0 = 0; g0 < n; g0 += 4) {
#pragma HLS PIPELINE II=4
            v8d vin0 = psi_in[(off0 + g0) >> 2];

            int gi0_0 = igk[g0 + 0] - 1;
            int gi0_1 = igk[g0 + 1] - 1;
            int gi0_2 = igk[g0 + 2] - 1;
            int gi0_3 = igk[g0 + 3] - 1;

            if ((unsigned)gi0_0 < (unsigned)Ngrid0) {
                int kS0 = gi0_0 / (nr1 * nr2);
                int rS0 = gi0_0 - kS0 * (nr1 * nr2);
                int jS0 = rS0 / nr1;
                int iS0 = rS0 - jS0 * nr1;
                psi_r_re[kS0][jS0][iS0] = vin0.d[0];
                psi_r_im[kS0][jS0][iS0] = vin0.d[1];
            }
            if ((unsigned)gi0_1 < (unsigned)Ngrid0) {
                int kS1 = gi0_1 / (nr1 * nr2);
                int rS1 = gi0_1 - kS1 * (nr1 * nr2);
                int jS1 = rS1 / nr1;
                int iS1 = rS1 - jS1 * nr1;
                psi_r_re[kS1][jS1][iS1] = vin0.d[2];
                psi_r_im[kS1][jS1][iS1] = vin0.d[3];
            }
            if ((unsigned)gi0_2 < (unsigned)Ngrid0) {
                int kS2 = gi0_2 / (nr1 * nr2);
                int rS2 = gi0_2 - kS2 * (nr1 * nr2);
                int jS2 = rS2 / nr1;
                int iS2 = rS2 - jS2 * nr1;
                psi_r_re[kS2][jS2][iS2] = vin0.d[4];
                psi_r_im[kS2][jS2][iS2] = vin0.d[5];
            }
            if ((unsigned)gi0_3 < (unsigned)Ngrid0) {
                int kS3 = gi0_3 / (nr1 * nr2);
                int rS3 = gi0_3 - kS3 * (nr1 * nr2);
                int jS3 = rS3 / nr1;
                int iS3 = rS3 - jS3 * nr1;
                psi_r_re[kS3][jS3][iS3] = vin0.d[6];
                psi_r_im[kS3][jS3][iS3] = vin0.d[7];
            }
        }

        // forward
        z_stage_3d_fwd(psi_r_re, psi_r_im, bufA_re, bufA_im, nr1, nr2, nr3, lb_s0, tmp_re_s0, tmp_im_s0);
        y_stage_3d_fwd(bufA_re,  bufA_im,  bufB_re, bufB_im, nr1, nr2, nr3, lb_s0, tmp_re_s0, tmp_im_s0);
        x_fused_stage_3d(bufB_re, bufB_im, bufA_re, bufA_im, v_local, nr1, nr2, nr3, lb_s0, tmp_re_s0, tmp_im_s0);
        // inverse
        y_stage_3d_inv(bufA_re,  bufA_im,  bufB_re, bufB_im, nr1, nr2, nr3, lb_s0, tmp_re_s0, tmp_im_s0);
        z_stage_3d_inv(bufB_re,  bufB_im,  psi_r_re, psi_r_im, nr1, nr2, nr3, lb_s0, tmp_re_s0, tmp_im_s0);

gather0:
        for (int g1 = 0; g1 < n; g1 += 4) {
#pragma HLS PIPELINE II=4
            v8d vout0;

            int gi1_0 = igk[g1 + 0] - 1;
            int gi1_1 = igk[g1 + 1] - 1;
            int gi1_2 = igk[g1 + 2] - 1;
            int gi1_3 = igk[g1 + 3] - 1;

            double r0=0,i0=0,r1=0,i1=0,r2=0,i2=0,r3=0,i3=0;

            if ((unsigned)gi1_0 < (unsigned)Ngrid0) {
                int kG0 = gi1_0 / (nr1 * nr2);
                int rG0 = gi1_0 - kG0 * (nr1 * nr2);
                int jG0 = rG0 / nr1;
                int iG0 = rG0 - jG0 * nr1;
                r0 = psi_r_re[kG0][jG0][iG0];
                i0 = psi_r_im[kG0][jG0][iG0];
            }
            if ((unsigned)gi1_1 < (unsigned)Ngrid0) {
                int kG1 = gi1_1 / (nr1 * nr2);
                int rG1 = gi1_1 - kG1 * (nr1 * nr2);
                int jG1 = rG1 / nr1;
                int iG1 = rG1 - jG1 * nr1;
                r1 = psi_r_re[kG1][jG1][iG1];
                i1 = psi_r_im[kG1][jG1][iG1];
            }
            if ((unsigned)gi1_2 < (unsigned)Ngrid0) {
                int kG2 = gi1_2 / (nr1 * nr2);
                int rG2 = gi1_2 - kG2 * (nr1 * nr2);
                int jG2 = rG2 / nr1;
                int iG2 = rG2 - jG2 * nr1;
                r2 = psi_r_re[kG2][jG2][iG2];
                i2 = psi_r_im[kG2][jG2][iG2];
            }
            if ((unsigned)gi1_3 < (unsigned)Ngrid0) {
                int kG3 = gi1_3 / (nr1 * nr2);
                int rG3 = gi1_3 - kG3 * (nr1 * nr2);
                int jG3 = rG3 / nr1;
                int iG3 = rG3 - jG3 * nr1;
                r3 = psi_r_re[kG3][jG3][iG3];
                i3 = psi_r_im[kG3][jG3][iG3];
            }

            vout0.d[0]=r0; vout0.d[1]=i0;
            vout0.d[2]=r1; vout0.d[3]=i1;
            vout0.d[4]=r2; vout0.d[5]=i2;
            vout0.d[6]=r3; vout0.d[7]=i3;

            hpsi_out[(off0 + g1) >> 2] = vout0;
        }
    }
}

