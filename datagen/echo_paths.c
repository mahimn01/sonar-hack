/* echo_paths.c — the echo-path loop of the renderer as ONE fused GPU kernel.
 *
 * Plain words: render_torch.synth_paths_torch adds up the echo of every body part with about 300
 * separate PyTorch operations per batch, each of which reads and writes hundreds of megabytes. Here
 * the same arithmetic runs in one pass: one GPU thread per (clip, baseband sample) walks over that
 * clip's reflectors, works out the 8 echo paths of each (2 speakers x 2 microphones x direct / via
 * the desk) and adds them up in registers. Nothing is written but the final sums.
 *
 * It is the same maths in the same order as synth_paths_torch (and render.py): path lengths, read
 * position and carrier phase in double precision, amplitudes and the sum in single precision.
 *
 * One source, two uses:
 *   - compiled at run time for the GPU by NVRTC (echo_kernel.py), with -DSONAR_CUDA;
 *   - compiled as ordinary C for the tests, which compare it with the PyTorch path on a CPU.
 * Read-only: it writes its two output arrays and nothing else.
 */
#ifdef SONAR_CUDA
#define DEV __device__ __forceinline__
#else
#include <math.h>
#define DEV static inline
#endif

/* how well a speaker / microphone couples in this direction (render._directivity) */
DEV float directivity(float cos_t, float n, float fl)
{
    if (n > 0.0f) {
        float b = cos_t > 0.0f ? cos_t : 0.0f;
        return fl + (1.0f - fl) * powf(b, n);
    }
    return 1.0f;
}

/* distance from a reflector to a fixed point (exact), unit vector towards it, and the range loss */
DEV void leg(const double *pt, double px, double py, double pz, float gamma, float kappa,
             double *r_out, float *u, float *loss)
{
    double dx = pt[0] - px, dy = pt[1] - py, dz = pt[2] - pz;
    double r = sqrt(dx * dx + dy * dy + dz * dz);
    if (r < 0.03) r = 0.03;
    float r32 = (float)r;
    u[0] = (float)dx / r32;
    u[1] = (float)dy / r32;
    u[2] = (float)dz / r32;
    *loss = powf(r32, -gamma) * expf(-kappa * r32);
    *r_out = r;
}

/* Everything for one (clip, baseband sample): the sum over that clip's reflectors and their paths.
 *
 * Per clip:      cd [n_clip, 14] double: speakers (2 x xyz), microphones (2 x xyz), speed of sound, read offset
 *                cf [n_clip, 19] float : speaker axes (2 x xyz), microphone axis (xyz), spk_n, spk_floor, mic_n,
 *                                        mic_floor, table_refl, k_amp, gamma, kappa, speaker 0 on, speaker 1 on
 *                ci [n_clip, 3]  int   : row of the clip's speaker envelopes, first reflector, one past the last
 * Per reflector: pos_c [n, n_piece, 4, 3] double, rho_c [n, n_piece, 4] float, nrm_c [n, n_piece, 4, 3] float:
 *                cubic spline pieces (highest power first), sf [n, 2] float: mirror exponent and floor,
 *                flags [n] int: bit 0 = mirror-like, bit 1 = part of the labelled motion
 * Per sample:    piece [nb] int (which spline piece), dt [nb] double (time into it), kdec [nb] double
 * env [rows, 2, n_total, 2] float: complex envelope of each speaker signal
 * dyn, key [n_clip, n_mic, nb, 2] float: the results (real, imaginary)
 */
DEV void echo_point(int c, int k, int nb, int n_mic, int n_piece, long long n_total, double fs, double f0,
                    const double *cd, const float *cf, const int *ci,
                    const double *pos_c, const float *rho_c, const float *nrm_c, const float *sf, const int *flags,
                    const int *piece, const double *dt, const double *kdec, const float *env,
                    float *dyn, float *key)
{
    const double *d_ = cd + (long long)c * 14;
    const float *f_ = cf + (long long)c * 19;
    const int *i_ = ci + (long long)c * 3;
    const double sound = d_[12], base = kdec[k] + d_[13];
    const float mic_axis0 = f_[6], mic_axis1 = f_[7], mic_axis2 = f_[8];
    const float spk_n = f_[9], spk_fl = f_[10], mic_n = f_[11], mic_fl = f_[12], table = f_[13];
    const float k_amp = f_[14], gamma = f_[15], kappa = f_[16];
    const long long env_row = i_[0];
    const int ip = piece[k];
    const double t = dt[k];
    const float tf = (float)t;
    float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};          /* [microphone][re, im] */
    float ksum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int j = i_[1]; j < i_[2]; ++j) {
        const double *pc = pos_c + ((long long)j * n_piece + ip) * 12;
        const double px = ((pc[0] * t + pc[3]) * t + pc[6]) * t + pc[9];
        const double py = ((pc[1] * t + pc[4]) * t + pc[7]) * t + pc[10];
        const double pz = ((pc[2] * t + pc[5]) * t + pc[8]) * t + pc[11];
        const float *rc = rho_c + ((long long)j * n_piece + ip) * 4;
        const float rho = ((rc[0] * tf + rc[1]) * tf + rc[2]) * tf + rc[3];
        const int spec = flags[j] & 1;
        float nx = 0.0f, ny = 0.0f, nz = 0.0f;
        if (spec) {
            const float *nc = nrm_c + ((long long)j * n_piece + ip) * 12;
            nx = ((nc[0] * tf + nc[3]) * tf + nc[6]) * tf + nc[9];
            ny = ((nc[1] * tf + nc[4]) * tf + nc[7]) * tf + nc[10];
            nz = ((nc[2] * tf + nc[5]) * tf + nc[8]) * tf + nc[11];
        }
        const float spec_q = sf[(long long)j * 2], spec_fl = sf[(long long)j * 2 + 1];

        /* reflector -> each speaker */
        double r_tx[2];
        float u_tx[2][3], g_tx[2];
        for (int s = 0; s < 2; ++s) {
            float loss;
            leg(d_ + 3 * s, px, py, pz, gamma, kappa, &r_tx[s], u_tx[s], &loss);
            const float *ax = f_ + 3 * s;
            float g = directivity(-(u_tx[s][0] * ax[0] + u_tx[s][1] * ax[1] + u_tx[s][2] * ax[2]), spk_n, spk_fl);
            g_tx[s] = g * loss * f_[17 + s];
        }
        const float base_amp = k_amp * rho;

        for (int m = 0; m < n_mic; ++m) {
            const double *mic = d_ + 6 + 3 * m;
            double mirror[3];
            mirror[0] = mic[0];
            mirror[1] = mic[1];
            mirror[2] = -mic[2];                               /* the same microphone mirrored in the desk */
            double r_rx[2];
            float u_rx[2][3], g_rx[2], loss;
            leg(mic, px, py, pz, gamma, kappa, &r_rx[0], u_rx[0], &loss);
            const float g_mic = directivity(-(u_rx[0][0] * mic_axis0 + u_rx[0][1] * mic_axis1 + u_rx[0][2] * mic_axis2), mic_n, mic_fl);
            g_rx[0] = g_mic * loss;
            leg(mirror, px, py, pz, gamma, kappa, &r_rx[1], u_rx[1], &loss);
            g_rx[1] = g_mic * loss * table;

            float a_re = 0.0f, a_im = 0.0f;
            for (int v = 0; v < 2; ++v) {
                for (int s = 0; s < 2; ++s) {
                    float amp = base_amp * g_tx[s] * g_rx[v];
                    if (spec) {                                /* mirror-like parts: palm, chest */
                        float bx = u_tx[s][0] + u_rx[v][0], by = u_tx[s][1] + u_rx[v][1], bz = u_tx[s][2] + u_rx[v][2];
                        float bn = sqrtf(bx * bx + by * by + bz * bz);
                        if (bn < 1e-9f) bn = 1e-9f;
                        float cosb = fabsf(nx * (bx / bn) + ny * (by / bn) + nz * (bz / bn));
                        amp = amp * (spec_fl + (1.0f - spec_fl) * powf(cosb, spec_q));
                    }
                    /* render.add_path: exact carrier phase, linearly interpolated envelope */
                    const double tau = (r_tx[s] + r_rx[v]) / sound;
                    const double idx = base - tau * fs;
                    const double i0f = floor(idx);
                    const float frac = (float)(idx - i0f);
                    long long i0 = (long long)i0f;
                    if (i0 < 0) i0 = 0;
                    if (i0 > n_total - 2) i0 = n_total - 2;
                    const float *e = env + ((env_row * 2 + s) * n_total + i0) * 2;
                    const float e_re = e[0] * (1.0f - frac) + e[2] * frac;
                    const float e_im = e[1] * (1.0f - frac) + e[3] * frac;
                    const double cyc = f0 * tau;
                    const float ph = (float)(cyc - floor(cyc)) * -6.283185307179586f;
                    const float cr = amp * cosf(ph), si = amp * sinf(ph);
                    a_re += cr * e_re - si * e_im;
                    a_im += cr * e_im + si * e_re;
                }
            }
            sum[m][0] += a_re;
            sum[m][1] += a_im;
            if (flags[j] & 2) {
                ksum[m][0] += a_re;
                ksum[m][1] += a_im;
            }
        }
    }
    for (int m = 0; m < n_mic; ++m) {
        const long long o = (((long long)c * n_mic + m) * nb + k) * 2;
        dyn[o] = sum[m][0];
        dyn[o + 1] = sum[m][1];
        key[o] = ksum[m][0];
        key[o + 1] = ksum[m][1];
    }
}

#ifdef SONAR_CUDA
extern "C" __global__ void echo_paths(int n_clip, int nb, int n_mic, int n_piece, long long n_total, double fs, double f0,
                                      const double *cd, const float *cf, const int *ci,
                                      const double *pos_c, const float *rho_c, const float *nrm_c, const float *sf, const int *flags,
                                      const int *piece, const double *dt, const double *kdec, const float *env,
                                      float *dyn, float *key)
{
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long long)n_clip * nb) return;
    echo_point((int)(id / nb), (int)(id % nb), nb, n_mic, n_piece, n_total, fs, f0, cd, cf, ci,
               pos_c, rho_c, nrm_c, sf, flags, piece, dt, kdec, env, dyn, key);
}
#else
/* the same loop on a CPU, one (clip, sample) after the other: only for the tests */
void echo_paths(int n_clip, int nb, int n_mic, int n_piece, long long n_total, double fs, double f0,
                const double *cd, const float *cf, const int *ci,
                const double *pos_c, const float *rho_c, const float *nrm_c, const float *sf, const int *flags,
                const int *piece, const double *dt, const double *kdec, const float *env,
                float *dyn, float *key)
{
    for (int c = 0; c < n_clip; ++c)
        for (int k = 0; k < nb; ++k)
            echo_point(c, k, nb, n_mic, n_piece, n_total, fs, f0, cd, cf, ci, pos_c, rho_c, nrm_c, sf, flags, piece, dt, kdec, env, dyn, key);
}
#endif
