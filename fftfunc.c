#include <stdint.h>

#define NX 8
#define NY 8
#define NZ 8
#define N  (NX*NY*NZ)

#define LANES 8

typedef struct { uint64_t lane[8]; } wide_t;
typedef struct { float re; float im; } cplx;
typedef union  { uint64_t u; float f[2]; } pack64;

const float INV_SQRT2 = 0.70710678118f;

static inline void fft8_core(const cplx in[8], cplx out[8]) {
#pragma HLS INLINE
    cplx t[8], v[8];
#pragma HLS ARRAY_PARTITION variable=t complete
#pragma HLS ARRAY_PARTITION variable=v complete

    for (int i0=0;i0<4;i0++) {
#pragma HLS UNROLL
        t[i0].re   = in[i0].re   + in[i0+4].re;
        t[i0].im   = in[i0].im   + in[i0+4].im;
        t[i0+4].re = in[i0].re   - in[i0+4].re;
        t[i0+4].im = in[i0].im   - in[i0+4].im;
    }

    v[0].re = t[0].re + t[2].re;  v[0].im = t[0].im + t[2].im;
    v[1].re = t[1].re + t[3].re;  v[1].im = t[1].im + t[3].im;
    v[2].re = t[0].re - t[2].re;  v[2].im = t[0].im - t[2].im;
    v[3].re = t[1].im - t[3].im;  v[3].im = t[3].re - t[1].re;

    out[0].re = v[0].re + v[1].re;  out[0].im = v[0].im + v[1].im;
    out[4].re = v[0].re - v[1].re;  out[4].im = v[0].im - v[1].im;
    out[2].re = v[2].re + v[3].re;  out[2].im = v[2].im + v[3].im;
    out[6].re = v[2].re - v[3].re;  out[6].im = v[2].im - v[3].im;

    cplx t5 = {(t[5].re + t[5].im)*INV_SQRT2,
               (t[5].im - t[5].re)*INV_SQRT2};
    cplx t6 = { t[6].im, -t[6].re };
    cplx t7 = {(t[7].im - t[7].re)*INV_SQRT2,
               (-t[7].re - t[7].im)*INV_SQRT2};

    cplx a = {t[4].re + t6.re, t[4].im + t6.im};
    cplx b = {t5.re + t7.re, t5.im + t7.im};
    cplx d = {t[4].re - t6.re, t[4].im - t6.im};
    cplx e = {t5.re - t7.re, t5.im - t7.im};

    out[1].re = a.re + b.re;  out[1].im = a.im + b.im;
    out[5].re = a.re - b.re;  out[5].im = a.im - b.im;
    out[3].re = d.re + e.im;  out[3].im = d.im - e.re;
    out[7].re = d.re - e.im;  out[7].im = d.im + e.re;
}

void fft3d_forward_hls(wide_t *gmem) {
#pragma HLS INTERFACE m_axi     port=gmem offset=slave bundle=gmem depth=64
#pragma HLS INTERFACE s_axilite port=gmem bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    cplx b0_0[N];
    cplx b1_0[N];
    cplx b2_0[NZ][NX][NY];
    cplx b3_0[NZ][NX][NY];

#pragma HLS BIND_STORAGE variable=b0_0 type=RAM_2P impl=bram
#pragma HLS BIND_STORAGE variable=b1_0 type=RAM_2P impl=bram
#pragma HLS BIND_STORAGE variable=b2_0 type=RAM_2P impl=bram
#pragma HLS BIND_STORAGE variable=b3_0 type=RAM_2P impl=bram

#pragma HLS ARRAY_PARTITION variable=b0_0 cyclic factor=64 dim=1
#pragma HLS ARRAY_PARTITION variable=b1_0 cyclic factor=64 dim=1

#pragma HLS ARRAY_PARTITION variable=b2_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=b2_0 complete dim=3
#pragma HLS ARRAY_PARTITION variable=b3_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=b3_0 complete dim=3

    // Scratch buffers (single set)
    cplx lin [LANES][8];
    cplx lout[LANES][8];

#pragma HLS ARRAY_PARTITION variable=lin  complete dim=1
#pragma HLS ARRAY_PARTITION variable=lout complete dim=1
#pragma HLS ARRAY_PARTITION variable=lin  complete dim=2
#pragma HLS ARRAY_PARTITION variable=lout complete dim=2

load0:
    for (int m0=0; m0<64; m0++) {
#pragma HLS PIPELINE II=2
        wide_t w0 = gmem[m0];
        for (int k0=0; k0<8; k0++) {
#pragma HLS UNROLL
            pack64 p0; p0.u = w0.lane[k0];
            b0_0[m0*8+k0].re = p0.f[0];
            b0_0[m0*8+k0].im = p0.f[1];
        }
    }

    for (int rep=0; rep<1000; rep++) {

loop_x:
        for (int i0=0; i0<64; i0 += LANES) {
#pragma HLS PIPELINE II=1
            for (int l0=0; l0<LANES; l0++) {
#pragma HLS UNROLL factor=2
                int base = (i0 + l0)*8;

                for (int x0=0; x0<8; x0++) {
                    lin[l0][x0] = b0_0[base + x0];
                }

                fft8_core(lin[l0], lout[l0]);

                for (int x1=0; x1<8; x1++) {
                    b1_0[base + x1] = lout[l0][x1];
                }
            }
        }

loop_y:
        for (int i1=0; i1<64; i1 += LANES) {
#pragma HLS PIPELINE II=1
            for (int l1=0; l1<LANES; l1++) {
#pragma HLS UNROLL factor=2
                int i  = i1 + l1;
                int z1 = i >> 3;
                int x1 = i & 7;

                for (int y0=0; y0<8; y0++) {
                    lin[l1][y0] = b1_0[z1*64 + y0*8 + x1];
                }

                fft8_core(lin[l1], lout[l1]);

                for (int y1=0; y1<8; y1++) {
                    b2_0[z1][x1][y1] = lout[l1][y1];
                }
            }
        }

loop_z:
        for (int i2=0; i2<64; i2 += LANES) {
#pragma HLS PIPELINE II=1
            for (int l2=0; l2<LANES; l2++) {
#pragma HLS UNROLL factor=2
                int i  = i2 + l2;
                int y2 = i >> 3;
                int x2 = i & 7;

                for (int z0=0; z0<8; z0++) {
                    lin[l2][z0] = b2_0[z0][x2][y2];
                }

                fft8_core(lin[l2], lout[l2]);

                for (int z1=0; z1<8; z1++) {
                    b3_0[z1][x2][y2] = lout[l2][z1];
                }
            }
        }
    }

store0:
    for (int m=0; m<64; m++) {
#pragma HLS PIPELINE II=2
        wide_t w;
        int z = m >> 3;
        int y = m & 7;
        for (int x=0; x<8; x++) {
#pragma HLS UNROLL
            pack64 p;
            p.f[0] = b3_0[z][x][y].re;
            p.f[1] = b3_0[z][x][y].im;
            w.lane[x] = p.u;
        }
        gmem[m] = w;
    }
}
