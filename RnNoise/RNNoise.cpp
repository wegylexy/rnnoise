#define _USE_MATH_DEFINES
#include "RNNoise.h"
#include "rnn_weight.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#define VAD_GRU_SIZE 24
#define NOISE_GRU_SIZE 48
#define DENOISE_GRU_SIZE 96

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120 << FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2 * FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)

#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD + PITCH_FRAME_SIZE)

#define NB_BANDS 22

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS + 3 * NB_DELTA_CEPS + 2)

static const auto WEIGHTS_SCALE = 1 / 256.f;

#define MAX_NEURONS 128

#define ACTIVATION_TANH    0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_RELU    2

#define INPUT_DENSE_SIZE 24

#define VAD_GRU_SIZE 24

#define NOISE_GRU_SIZE 48

#define DENOISE_GRU_SIZE 96

static constexpr int16_t eband5ms[] = {
	// 0 200 400 600 800 1k 1.2 1.4 1.6 2k 2.4 2.8 3.2 4k 4.8 5.6 6.8 8k 9.6 12k 15.6 20k
	0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};

using namespace std;

struct RNNState {
	float vad_gru_state[VAD_GRU_SIZE], noise_gru_state[NOISE_GRU_SIZE], denoise_gru_state[DENOISE_GRU_SIZE];
};

struct DenoiseState {
	float analysis_mem[FRAME_SIZE], cepstral_mem[CEPS_MEM][NB_BANDS];
	int memid;
	float synthesis_mem[FRAME_SIZE], pitch_buf[PITCH_BUF_SIZE], pitch_enh_buf[PITCH_BUF_SIZE], last_gain;
	int last_period;
	float mem_hp_x[2], lastg[NB_BANDS];
	RNNState rnn;
	float yy_lookup[(PITCH_MAX_PERIOD >> 1) + 1];
};

#define kiss_fft_scalar float
#define kiss_twiddle_scalar float
typedef std::complex<kiss_fft_scalar> kiss_fft_cpx;
typedef std::complex<kiss_twiddle_scalar> kiss_twiddle_cpx;

struct DenseLayer {
	const rnn_weight *bias, *input_weights;
	int nb_inputs, nb_neurons, activation;
};

struct GRULayer {
	const rnn_weight *bias, *input_wieghts, *recurrent_weights;
	int nb_inputs, nb_neurons, activation;
};

#define MAXFACTORS 8

inline static void compute_twiddles(kiss_twiddle_cpx *twiddles, const int nfft) {
	const auto _ = -2 * static_cast<kiss_twiddle_scalar>(M_PI) / nfft;
	for (int i{}; i < nfft; ++i) {
		auto &twiddle = twiddles[i];
		const auto phase = _ * i;
		twiddle.real(cos(phase));
		twiddle.imag(sin(phase));
	}
}

inline static void kf_factor(const int n, int16_t facbuf[2 * MAXFACTORS]) {
	int p{ 4 }, stages{}, m{ n };
	do {
		while (m % p) {
			switch (p) {
			case 4:
				p = 2;
				break;
			case 2:
				p = 3;
				break;
			default:
				p += 2;
				break;
			}
			if (p > 32000 || p * p > m)
				p = m;
		}
		m /= p;
#ifdef RADIX_TWO_ONLY
		if (p != 2 && p != 4)
#else
		if (p > 5)
#endif
			throw invalid_argument{ "Invalid numbers of factors" };
		facbuf[2 * stages] = p;
		if (p == 2 && stages > 1) {
			facbuf[2 * stages] = 4;
			facbuf[2] = 2;
		}
		++stages;
	} while (m > 1);
	m = n;
	int stages_2{ stages / 2 }, i;
	for (i = 0; i < stages_2; ++i)
		swap(facbuf[2 * i], facbuf[2 * (stages - i - 1)]);
	for (i = 0; i < stages; ++i)
		facbuf[2 * i + 1] = m /= facbuf[2 * i];
}

struct kiss_fft_state;
static void compute_bitrev_table(int Fout, int16_t *f, const size_t fstride, int in_stride, int16_t *factors, const kiss_fft_state *st) {
	const int p{ *factors++ }, m{ *factors++ };
	if (m == 1) {
		for (int j{}; j < p; ++j) {
			*f = Fout + j;
			f += fstride * in_stride;
		}
	}
	else {
		for (int j{}; j < p; ++j) {
			compute_bitrev_table(Fout, f, fstride * p, in_stride, factors, st);
			f += fstride * in_stride;
			Fout += m;
		}
	}
}

struct kiss_fft_state {
	int nfft;
	float scale;
	int shift;
	int16_t factors[2 * MAXFACTORS];
	int16_t *bitrev;
	kiss_twiddle_cpx *twiddles;

	kiss_fft_state(const int nfft) :
		nfft{ nfft },
		scale{ 1.f / nfft },
		shift{ -1 },
		factors{},
		bitrev{ new int16_t[nfft] },
		twiddles{ new kiss_twiddle_cpx[nfft] }
	{
		compute_twiddles(twiddles, nfft);
		kf_factor(nfft, factors);
		compute_bitrev_table(0, bitrev, 1, 1, factors, this);
	}

	~kiss_fft_state() {
		delete[] twiddles;
		delete[] bitrev;
	}
};

static const struct CommonState {
	const kiss_fft_state &kfft;
	float half_window[FRAME_SIZE], dct_table[NB_BANDS * NB_BANDS];

	CommonState() : kfft{ *new kiss_fft_state{WINDOW_SIZE} } {
		static const float __2{ sqrt(.5f) }, M_PI_F{ static_cast<float>(M_PI) };
		int i;
		for (i = 0; i < FRAME_SIZE; ++i) {
			const auto t = sin(M_PI_2 * (i + .5) / FRAME_SIZE);
			half_window[i] = static_cast<float>(sin(M_PI_2 * t * t));
		}
		for (i = 0; i < NB_BANDS; ++i) {
			for (int j = 0; j < NB_BANDS; ++j) {
				dct_table[i * NB_BANDS + j] = cos((i + .5f) * j * M_PI_F / NB_BANDS);
				if (!j)
					dct_table[i * NB_BANDS] *= __2;
			}
		}
	}

	~CommonState() { delete &kfft; }
} common{};

static void biquad(float y[FRAME_SIZE], float mem[2], const float x[FRAME_SIZE], const float b[2], const float a[2], const int N) {
	for (auto i = 0; i < N; ++i) {
		float xi{ x[i] * 32767.f }, yi{ y[i] = xi + mem[0] };
		mem[0] = b[0] * xi - a[0] * yi + mem[1];
		mem[1] = b[1] * xi - a[1] * yi;
	}
}

inline static void kf_bfly2(kiss_fft_cpx *Fout, int m, int N) {
	kiss_fft_cpx *Fout2;
	if (m == 1) {
		for (int i{}; i < N; ++i, Fout += 2) {
			Fout2 = Fout + 1;
			const auto t = *Fout2;
			*Fout2 = *Fout - t;
			*Fout += t;
		}
	}
	else {
		static const float tw{ sqrt(.5f) };
		for (int i{}; i < N; ++i, Fout += 5) {
			Fout2 = Fout + 4;
			auto t = *Fout2;
			*Fout2 = *Fout - t;
			*Fout += t;
			++Fout;
			++Fout2;
			t.real((Fout2->real() + Fout2->imag()) * tw);
			t.imag((Fout2->imag() - Fout2->real()) * tw);
			*Fout2 = *Fout - t;
			*Fout += t;
			++Fout;
			++Fout2;
			t.real(Fout2->imag());
			t.imag(-Fout2->real());
			*Fout2 = *Fout - t;
			*Fout += t;
			++Fout;
			++Fout2;
			t.real((Fout2->imag() - Fout2->real()) * tw);
			t.imag(-(Fout2->imag() + Fout2->real()) * tw);
			*Fout2 = *Fout - t;
			*Fout += t;
		}
	}
}

inline static void kf_bfly4(kiss_fft_cpx *Fout, const size_t fstride, const kiss_fft_state &st, int m, int N, int mm) {
	if (m == 1) {
		for (int i{}; i < N; ++i, Fout += 4) {
			kiss_fft_cpx scratch0{ *Fout - Fout[2] };
			*Fout += Fout[2];
			kiss_fft_cpx scratch1{ Fout[1] + Fout[3] };
			Fout[2] = *Fout - scratch1;
			*Fout += scratch1;
			scratch1 = Fout[1] - Fout[3];
			Fout[1].real(scratch0.real() + scratch1.imag());
			Fout[1].imag(scratch0.imag() - scratch1.real());
			Fout[3].real(scratch0.real() - scratch1.imag());
			Fout[3].imag(scratch0.imag() + scratch1.real());
		}
	}
	else {
		const kiss_twiddle_cpx *tw1, *tw2, *tw3;
		const int m2{ 2 * m }, m3{ 3 * m };
		kiss_fft_cpx scratch[6], *Fout_beg = Fout;
		for (int i{}; i < N; ++i) {
			Fout = Fout_beg + i * mm;
			tw3 = tw2 = tw1 = st.twiddles;
			for (int j{}; j < m; ++j, ++Fout) {
				scratch[0] = Fout[m] * *tw1;
				scratch[1] = Fout[m2] * *tw2;
				scratch[2] = Fout[m3] * *tw3;
				scratch[5] = *Fout - scratch[1];
				*Fout += scratch[1];
				scratch[3] = scratch[0] + scratch[2];
				scratch[4] = scratch[0] - scratch[2];
				Fout[m2] = *Fout - scratch[3];
				tw1 += fstride;
				tw2 += fstride * 2;
				tw3 += fstride * 3;
				*Fout += scratch[3];
				Fout[m].real(scratch[5].real() + scratch[4].imag());
				Fout[m].imag(scratch[5].imag() - scratch[4].real());
				Fout[m3].real(scratch[5].real() - scratch[4].imag());
				Fout[m3].imag(scratch[5].imag() + scratch[4].real());
			}
		}
	}
}

#ifndef RADIS_TWO_ONLY
inline static void kf_bfly3(kiss_fft_cpx *Fout, const size_t fstride, const kiss_fft_state &st, int m, int N, int mm) {
	const size_t m2{ 2 * static_cast<size_t>(m) };
	kiss_fft_cpx scratch[5];
	kiss_fft_cpx *Fout_beg{ Fout };
	const kiss_twiddle_cpx *tw1, *tw2, epi3{ st.twiddles[fstride * m] };
	for (int i{}; i < N; ++i) {
		Fout = Fout_beg + i * mm;
		tw2 = tw1 = st.twiddles;
		auto k = m;
		do {
			scratch[1] = Fout[m] * *tw1;
			scratch[2] = Fout[m2] * *tw2;
			scratch[3] = scratch[1] + scratch[2];
			scratch[0] = scratch[1] - scratch[2];
			tw1 += fstride;
			tw2 += fstride * 2;
			Fout[m].real(Fout->real() - scratch[3].real() * .5f);
			Fout[m].imag(Fout->imag() - scratch[3].imag() * .5f);
			scratch[0] *= epi3.imag();
			*Fout += scratch[3];
			Fout[m2].real(Fout[m].real() + scratch[0].imag());
			Fout[m2].imag(Fout[m].imag() - scratch[0].real());
			Fout[m].real(Fout[m].real() - scratch[0].imag());
			Fout[m].imag(Fout[m].imag() + scratch[0].real());
			++Fout;
		} while (--k);
	}
}

#ifndef OVERRIDE_kf_bfly5
inline static void kf_bfly5(kiss_fft_cpx *Fout, const size_t fstride, const kiss_fft_state &st, int m, int N, int mm) {
	const auto tw = st.twiddles;
	kiss_twiddle_cpx *Fout1, *Fout2, *Fout3, *Fout4, *Fout_beg{ Fout }, scratch[13],
		ya{ tw[fstride * m] }, yb{ tw[fstride * 2 * m] };
	for (int i{}; i < N; ++i) {
		Fout4 = (Fout3 = (Fout2 = (Fout1 = (Fout = Fout_beg + i * mm) + m) + m) + m) + m;
		for (int u{}; u < m; ++u, ++Fout, ++Fout1, ++Fout2, ++Fout3, ++Fout4) {
			scratch[0] = *Fout;
			scratch[1] = *Fout1 * tw[u * fstride];
			scratch[2] = *Fout2 * tw[2 * u * fstride];
			scratch[3] = *Fout3 * tw[3 * u * fstride];
			scratch[4] = *Fout4 * tw[4 * u * fstride];
			scratch[7] = scratch[1] + scratch[4];
			scratch[10] = scratch[1] - scratch[4];
			scratch[8] = scratch[2] + scratch[3];
			scratch[9] = scratch[2] - scratch[3];
			*Fout += scratch[7] + scratch[8];
			scratch[5] = scratch[7] * ya.real() + scratch[8] * yb.real() + scratch[0];
			auto s = scratch[10] * ya.imag() + scratch[9] * yb.imag();
			scratch[6].real(s.imag());
			scratch[6].imag(-s.real());
			*Fout1 = scratch[5] - scratch[6];
			*Fout4 = scratch[5] + scratch[6];
			scratch[11] = scratch[7] * yb.real() + scratch[8] * ya.real() + scratch[0];
			s = scratch[10] * yb.imag() - scratch[9] * ya.imag();
			scratch[12].real(-s.imag());
			scratch[12].imag(s.real());
			*Fout2 = scratch[11] + scratch[12];
			*Fout3 = scratch[11] - scratch[12];
		}
	}
}
#endif
#endif

inline static void opus_fft_impl(const kiss_fft_state &st, kiss_fft_cpx fout[WINDOW_SIZE]) {
	int m2, m, p, L{},
		fstride[MAXFACTORS],
		shift{ st.shift > 0 ? st.shift : 0 };
	fstride[0] = 1;
	do {
		p = st.factors[2 * L];
		m = st.factors[2 * L + 1];
		fstride[L + 1] = fstride[L] * p;
		++L;
	} while (m != 1);
	m = st.factors[2 * L - 1];
	for (int i{ L - 1 }; i >= 0; --i) {
		m2 = i ? st.factors[2 * i - 1] : 1;
		switch (st.factors[2 * i]) {
		case 2:
			kf_bfly2(fout, m, fstride[i]);
			break;
		case 4:
			kf_bfly4(fout, fstride[i] << shift, st, m, fstride[i], m2);
			break;
#ifndef RADIS_TWO_ONLY
		case 3:
			kf_bfly3(fout, fstride[i] << shift, st, m, fstride[i], m2);
			break;
		case 5:
			kf_bfly5(fout, fstride[i] << shift, st, m, fstride[i], m2);
			break;
#endif
		}
		m = m2;
	}
}

inline static void opus_fft_c(const kiss_fft_state &st, const kiss_fft_cpx fin[WINDOW_SIZE], kiss_fft_cpx fout[WINDOW_SIZE]) {
	float scale{ st.scale };
	for (int i{}; i < st.nfft; ++i)
		fout[st.bitrev[i]] = scale * fin[i];
	opus_fft_impl(st, fout);
}

static void forward_transform(kiss_fft_cpx out[FREQ_SIZE], const float in[WINDOW_SIZE]) {
	kiss_fft_cpx x[WINDOW_SIZE], y[WINDOW_SIZE];
	int i;
	for (i = 0; i < WINDOW_SIZE; ++i) {
		auto &_x = x[i];
		_x.real(in[i]);
		_x.imag(0);
	}
	opus_fft_c(common.kfft, x, y);
	memcpy(out, y, sizeof(kiss_fft_cpx) * FREQ_SIZE);
}

static void apply_window(float x[WINDOW_SIZE]) {
	for (int i{}; i < FRAME_SIZE; ++i) {
		const auto half_window = common.half_window[i];
		x[i] *= half_window;
		x[WINDOW_SIZE - 1 - i] *= half_window;
	}
}

static void compute_band_energy(float bandE[NB_BANDS], const kiss_fft_cpx X[FREQ_SIZE]) {
	memset(bandE, 0, sizeof(float) * NB_BANDS);
	int i;
	for (i = 0; i < NB_BANDS - 1; ++i) {
		const auto band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
		for (int j = 0; j < band_size; ++j) {
			const float frac{ static_cast<float>(j) / band_size },
				tmp{ norm(X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j]) };
			bandE[i] += (1 - frac) * tmp;
			bandE[i + 1] += frac * tmp;
		}
	}
	bandE[0] *= 2;
	bandE[NB_BANDS - 1] *= 2;
}

inline static void frame_analysis(float analysis_mem[FRAME_SIZE], kiss_fft_cpx X[FREQ_SIZE], float Ex[NB_BANDS], const float in[FRAME_SIZE]) {
	float x[WINDOW_SIZE];
	memcpy(x, analysis_mem, sizeof(float) * FRAME_SIZE);
	memcpy(x + FRAME_SIZE, in, sizeof(float) * FRAME_SIZE);
	memcpy(analysis_mem, in, sizeof(float) * FRAME_SIZE);
	apply_window(x);
	forward_transform(X, x);
	compute_band_energy(Ex, X);
}

inline static void xcorr_kernel(const float *x, const float *y, float sum[4], int len) {
	float y_0{ *y++ }, y_1{ *y++ }, y_2{ *y++ }, y_3, tmp;
	int j, len_3{ len - 3 };
	for (j = 0; j < len_3; j += 4) {
		tmp = *x++;
		y_3 = *y++;
		sum[0] += tmp * y_0;
		sum[1] += tmp * y_1;
		sum[2] += tmp * y_2;
		sum[3] += tmp * y_3;
		tmp = *x++;
		y_0 = *y++;
		sum[0] += tmp * y_1;
		sum[1] += tmp * y_2;
		sum[2] += tmp * y_3;
		sum[3] += tmp * y_0;
		tmp = *x++;
		y_1 = *y++;
		sum[0] += tmp * y_2;
		sum[1] += tmp * y_3;
		sum[2] += tmp * y_0;
		sum[3] += tmp * y_1;
		tmp = *x++;
		y_2 = *y++;
		sum[0] += tmp * y_3;
		sum[1] += tmp * y_0;
		sum[2] += tmp * y_1;
		sum[3] += tmp * y_2;
	}
	if (j++ < len) {
		tmp = *x++;
		y_3 = *y++;
		sum[0] += tmp * y_0;
		sum[1] += tmp * y_1;
		sum[2] += tmp * y_2;
		sum[3] += tmp * y_3;
	}
	if (j++ < len) {
		tmp = *x++;
		y_0 = *y++;
		sum[0] += tmp * y_1;
		sum[1] += tmp * y_2;
		sum[2] += tmp * y_3;
		sum[3] += tmp * y_0;
	}
	if (j++ < len) {
		tmp = *x++;
		y_1 = *y++;
		sum[0] += tmp * y_2;
		sum[1] += tmp * y_3;
		sum[2] += tmp * y_0;
		sum[3] += tmp * y_1;
	}
}

inline static float celt_inner_prod(const float *x, const float *y, int N) {
	float xy{};
	for (int i{}; i < N; ++i)
		xy += x[i] * y[i];
	return xy;
}

inline static void celt_pitch_xcorr(const float *_x, const float *_y, float *xcorr, int len, int max_pitch) {
	int i, max_pitch_3{ max_pitch - 3 };
	memset(xcorr, 0, sizeof(float) * max_pitch);
	for (i = 0; i < max_pitch_3; i += 4)
		xcorr_kernel(_x, _y + i, xcorr + i, len);
	for (; i < max_pitch; ++i)
		xcorr[i] = celt_inner_prod(_x, _y + i, len);
}

inline static void _celt_autocorr(const float *x, float *ac) {
	static constexpr int n{ PITCH_BUF_SIZE >> 1 };
	int i;
	const auto fastN = n - 4;
	celt_pitch_xcorr(x, x, ac, fastN, 4 + 1);
	for (int k{}; k <= 4; ++k) {
		float d{};
		for (i = k + fastN; i < n; ++i)
			d += x[i] * x[i - k];
		ac[k] += d;
	}
}

inline static void _celt_lpc(float *_lpc, const float *ac) {
	memset(_lpc, 0, sizeof(float) * 4);
	float error{ ac[0] };
	if (ac[0]) {
		for (int i{}; i < 4; ++i) {
			float rr{};
			for (int j{}; j < i; ++j)
				rr += _lpc[j] * ac[i - j];
			rr += ac[i + 1];
			const float r{ _lpc[i] = -rr / error };
			for (int j{}, i_1_2{ (i + 1) >> 1 }; j < i_1_2; ++j) {
				float tmp1{ _lpc[j] }, tmp2{ _lpc[i - 1 - j] };
				_lpc[j] += r * tmp2;
				_lpc[i - 1 - j] += r * tmp1;
			}
			if ((error -= r * r * error) < .001f * ac[0])
				break;
		}
	}
}

inline static void celt_fir5(const float *x, const float *num, float *y, int N, float *mem) {
	float num0{ num[0] }, num1{ num[1] }, num2{ num[2] }, num3{ num[3] }, num4{ num[4] },
		mem0{ mem[0] }, mem1{ mem[1] }, mem2{ mem[2] }, mem3{ mem[3] }, mem4{ mem[4] };
	for (int i{}; i < N; ++i) {
		float sum{ x[i] + num0 * mem0 + num1 * mem1 + num2 * mem2 + num3 * mem3 + num4 * mem4 };
		mem4 = mem3;
		mem3 = mem2;
		mem2 = mem1;
		mem1 = mem0;
		mem0 = x[i];
		y[i] = sum;
	}
	mem[0] = mem0;
	mem[1] = mem1;
	mem[2] = mem2;
	mem[3] = mem3;
	mem[4] = mem4;
}

inline static void pitch_downsample(float *const x[], float *x_lp) {
	static constexpr int len_2{ PITCH_BUF_SIZE >> 1 };
	int i;
	x_lp[0] = (x[0][1] * .5f + x[0][0])*.5f;
	for (i = 1; i < len_2; ++i)
		x_lp[i] = ((x[0][(2 * i - 1)] + x[0][(2 * i + 1)]) * .5f + x[0][2 * i]) * .5f;
	float ac[5], lpc[4];
	_celt_autocorr(x_lp, ac);
	ac[0] *= 1.0001f;
	for (i = 1; i < 5; ++i)
		ac[i] -= i * i * 6.4e-5f * ac[i];
	_celt_lpc(lpc, ac);
	float tmp{ 1 }, lpc2[5];
	for (i = 0; i < 4; ++i)
		lpc[i] *= tmp *= .9f;
	lpc2[0] = lpc[0] + .8f;
	lpc2[1] = lpc[1] + .8f * lpc[0];
	lpc2[2] = lpc[2] + .8f * lpc[1];
	lpc2[3] = lpc[3] + .8f * lpc[2];
	lpc2[4] = .8f * lpc[3];
	float mem[5]{};
	celt_fir5(x_lp, lpc2, x_lp, len_2, mem);
}

inline static void find_best_pitch(float *xcorr, float *y, int len, int max_pitch, int *best_pitch) {
	float Syy{ 1 }, best_num[]{ -1, -1 }, best_den[2]{};
	best_pitch[0] = 0;
	best_pitch[1] = 1;
	for (int j{}; j < len; ++j)
		Syy += y[j] * y[j];
	for (int i{}; i < max_pitch; ++i) {
		if (xcorr[i] > 0) {
			float xcorr16{ xcorr[i] }, num{ xcorr16 * xcorr16 };
			if (num * best_den[1] > best_num[1] * Syy) {
				if (num * best_den[0] > best_num[0] * Syy) {
					best_num[1] = best_num[0];
					best_den[1] = best_den[0];
					best_pitch[1] = best_pitch[0];
					best_num[0] = num;
					best_den[0] = Syy;
					best_pitch[0] = i;
				}
				else {
					best_num[1] = num;
					best_den[1] = Syy;
					best_pitch[1] = i;
				}
			}
		}
		const auto _y_ = y[i + len], _y = y[i];
		Syy = max(1.f, Syy + _y_ * _y_ - _y * _y);
	}
}

inline static void pitch_search(const float *x_lp, float *y, int &pitch) {
	static constexpr int max_pitch{ PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD };
	static constexpr int lag{ PITCH_FRAME_SIZE + max_pitch },
		len_4{ PITCH_FRAME_SIZE >> 2 }, lag_4{ lag >> 2 }, max_pitch_2{ max_pitch >> 1 }, max_pitch_4{ max_pitch >> 2 }, len_2{ PITCH_FRAME_SIZE >> 1 };
	int j, best_pitch[2]{}, offset;
	{
		float xcorr[max_pitch_2];
		{
			float y_lp4[lag_4];
			{
				float x_lp4[len_4];
				for (j = 0; j < len_4; ++j)
					x_lp4[j] = x_lp[2 * j];
				for (j = 0; j < lag_4; ++j)
					y_lp4[j] = y[2 * j];
				celt_pitch_xcorr(x_lp4, y_lp4, xcorr, len_4, max_pitch_4);
			}
			find_best_pitch(xcorr, y_lp4, len_4, max_pitch_4, best_pitch);
		}
		for (int i{}; i < max_pitch_2; ++i)
			xcorr[i] = abs(i - 2 * best_pitch[0]) > 2 && abs(i - 2 * best_pitch[1]) > 2 ? 0 : max(-1.f, celt_inner_prod(x_lp, y + i, len_2));
		find_best_pitch(xcorr, y, len_2, max_pitch_2, best_pitch);
		if (best_pitch[0] > 0 && best_pitch[0] < max_pitch_2 - 1) {
			float a{ xcorr[best_pitch[0] - 1] },
				b{ xcorr[best_pitch[0]] },
				c{ xcorr[best_pitch[0] + 1] };
			offset = c - a > .7f * (b - a) ? 1 : a - c > .7f * (b - c) ? -1 : 0;
		}
		else {
			offset = 0;
		}
	}
	pitch = 2 * best_pitch[0] - offset;
}

inline static void dual_inner_prod(const float *x, const float *y01, const float *y02, int N, float &xy1, float &xy2) {
	xy2 = xy1 = 0;
	for (int i{}; i < N; ++i) {
		xy1 += x[i] * y01[i];
		xy2 += x[i] * y02[i];
	}
}

inline static float compute_pitch_gain(float xy, float xx, float yy) {
	return xy / sqrt(1 + xx * yy);
}

static constexpr int second_check[] = { 0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2 };
inline static float remove_doubling(float *x, int &T0_, int prev_period, float prev_gain, float *yy_lookup) {
	static constexpr int maxperiod_2{ PITCH_MAX_PERIOD >> 1 }, minperiod_2{ PITCH_MIN_PERIOD >> 1 }, N_2{ PITCH_FRAME_SIZE >> 1 };
	T0_ /= 2;
	prev_period /= 2;
	x += maxperiod_2;
	if (T0_ >= maxperiod_2)
		T0_ = maxperiod_2 - 1;
	int T{ T0_ }, T0{ T }, k;
	float best_xy, best_yy, g;
	float xy, xx;
	dual_inner_prod(x, x, x - T0, N_2, xx, xy);
	float yy{ yy_lookup[0] = xx };
	for (int i = 1; i <= maxperiod_2; ++i) {
		const auto x_i = x[-i], x_N_i = x[N_2 - i];
		yy_lookup[i] = max(0.f, yy += x_i * x_i - x_N_i * x_N_i);
	}
	best_xy = xy;
	best_yy = yy = yy_lookup[T0];
	g = compute_pitch_gain(xy, xx, yy);
	float g0{ g };
	for (k = 2; k <= 15; ++k) {
		const auto T1 = (2 * T0 + k) / (2 * k);
		if (T1 < minperiod_2)
			break;
		int T1b;
		T1b = k == 2 ? T1 + T0 > maxperiod_2 ? T0 : T0 + T1 : (2 * second_check[k] * T0 + k) / (2 * k);
		float xy2;
		dual_inner_prod(x, x - T1, x - T1b, N_2, xy, xy2);
		xy = (xy + xy2) * .5f;
		yy = (yy_lookup[T1] + yy_lookup[T1b]) * .5f;
		auto g1 = compute_pitch_gain(xy, xx, yy);
		const auto T1_prev_period = abs(T1 - prev_period);
		float cont{ T1_prev_period <= 1 ? prev_gain : T1_prev_period <= 2 && 5 * k * k < T0 ? prev_gain * .5f : 0 };
		if (g1 > (T1 < 3 * minperiod_2 ? max(.4f, .85f * g0 - cont) : T1 < 2 * minperiod_2 ? max(.5f, .9f * g0 - cont) : max(.3f, .7f * g0 - cont))) {
			best_xy = xy;
			best_yy = yy;
			T = T1;
			g = g1;
		}
	}
	best_xy = max(0.f, best_xy);
	float pg{ best_yy <= best_xy ? 1.f : best_xy / (best_yy + 1) }, xcorr[3];
	for (k = 0; k < 3; ++k)
		xcorr[k] = celt_inner_prod(x, x - (T + k - 1), N_2);
	if ((T0_ = 2 * T + (xcorr[2] - xcorr[0] > .7f * (xcorr[1] - xcorr[0]) ? 1 : xcorr[0] - xcorr[2] > .7f * (xcorr[1] - xcorr[2]) ? -1 : 0)) < PITCH_MIN_PERIOD)
		T0_ = PITCH_MIN_PERIOD;
	return min(pg, g);
}

inline static void compute_band_corr(float bandE[NB_BANDS], const kiss_fft_cpx X[FREQ_SIZE], const kiss_fft_cpx P[FREQ_SIZE]) {
	memset(bandE, 0, sizeof(float) * NB_BANDS);
	for (int i{}; i < NB_BANDS - 1; ++i) {
		const auto band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
		for (int j{}; j < band_size; ++j) {
			const auto _j = (eband5ms[i] << FRAME_SIZE_SHIFT) + j;
			const auto &_X = X[_j], &_P = P[_j];
			float frac{ static_cast<float>(j) / band_size },
				tmp{ _X.real() * _P.real() + _X.imag() * _P.imag() };
			bandE[i] += (1 - frac) * tmp;
			bandE[i + 1] += frac * tmp;
		}
	}
	bandE[0] *= 2;
	bandE[NB_BANDS - 1] *= 2;
}

inline static void dct(float out[NB_BANDS], const float in[NB_BANDS]) {
	static const auto sqrt_11 = sqrt(1 / 11.f);
	for (int i{}; i < NB_BANDS; ++i) {
		float sum{};
		for (int j{}; j < NB_BANDS; ++j)
			sum += in[j] * common.dct_table[j * NB_BANDS + i];
		out[i] = sum * sqrt_11;
	}
}

inline static bool compute_frame_features(DenoiseState &st, kiss_fft_cpx X[FREQ_SIZE], kiss_fft_cpx P[WINDOW_SIZE],
	float Ex[NB_BANDS], float Ep[NB_BANDS], float Exp[NB_BANDS], float features[NB_FEATURES], const float in[FRAME_SIZE])
{
	int i;
	float pitch_buf[PITCH_BUF_SIZE >> 1], *const pre[1]{ st.pitch_buf };
	frame_analysis(st.analysis_mem, X, Ex, in);
	memmove(pre[0], pre[0] + FRAME_SIZE, sizeof(float) * (PITCH_BUF_SIZE - FRAME_SIZE));
	memcpy(pre[0] + PITCH_BUF_SIZE - FRAME_SIZE, in, sizeof(float) * FRAME_SIZE);
	pitch_downsample(pre, pitch_buf);
	int pitch_index;
	pitch_search(pitch_buf + (PITCH_MAX_PERIOD >> 1), pitch_buf, pitch_index);
	pitch_index = PITCH_MAX_PERIOD - pitch_index;
	st.last_gain = remove_doubling(pitch_buf, pitch_index, st.last_period, st.last_gain, st.yy_lookup);
	st.last_period = pitch_index;
	float p[WINDOW_SIZE], tmp[NB_BANDS], logMax{ -2 }, follow{ -2 }, Ly[NB_BANDS], E{};
	memcpy(p, pre[0] + PITCH_BUF_SIZE - WINDOW_SIZE - pitch_index, sizeof(float) * WINDOW_SIZE);
	apply_window(p);
	forward_transform(P, p);
	compute_band_energy(Ep, P);
	compute_band_corr(Exp, X, P);
	for (i = 0; i < NB_BANDS; ++i)
		Exp[i] /= sqrt(.001f + Ex[i] * Ep[i]);
	dct(tmp, Exp);
	memcpy(features + NB_BANDS + 2 * NB_DELTA_CEPS, tmp, sizeof(float) * NB_DELTA_CEPS);
	features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3f;
	features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= .9f;
	features[NB_BANDS + 3 * NB_DELTA_CEPS] = .01f * (pitch_index - 300);
	for (i = 0; i < NB_BANDS; ++i) {
		logMax = max(logMax, Ly[i] = max(logMax - 7, max(follow - 1.5f, log10(1e-2f + Ex[i]))));
		follow = max(follow - 1.5f, Ly[i]);
		E += Ex[i];
	}
	if (E < .04f) {
		memset(features, 0, sizeof(float) * NB_FEATURES);
		return true;
	}
	dct(features, Ly);
	features[0] -= 12;
	features[1] -= 4;
	float *ceps_0{ st.cepstral_mem[st.memid] },
		*ceps_1{ st.memid < 1 ? st.cepstral_mem[CEPS_MEM + st.memid - 1] : st.cepstral_mem[st.memid - 1] },
		*ceps_2{ st.memid < 2 ? st.cepstral_mem[CEPS_MEM + st.memid - 2] : st.cepstral_mem[st.memid - 2] },
		spec_variability{};
	memcpy(ceps_0, features, sizeof(float) * NB_BANDS);
	st.memid = (st.memid + 1) % CEPS_MEM;
	for (i = 0; i < NB_DELTA_CEPS; ++i) {
		features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
		features[NB_BANDS + i] = ceps_0[i] - ceps_2[i];
		features[NB_BANDS + NB_DELTA_CEPS + i] = ceps_0[i] - 2 * ceps_1[i] + ceps_2[i];
	}
	for (i = 0; i < CEPS_MEM; ++i) {
		float mindist{ 1e15f };
		for (int j{}; j < CEPS_MEM; ++j) {
			float dist{};
			for (int k{}; k < NB_BANDS; ++k) {
				const auto t = st.cepstral_mem[i][k] - st.cepstral_mem[j][k];
				dist += t * t;
			}
			if (j != i)
				mindist = min(mindist, dist);
		}
		spec_variability += mindist;
	}
	features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM - 2.1f;
	return false;
}

#define INPUT_SIZE 42

static constexpr DenseLayer input_dense{
	input_dense_bias,
	input_dense_weights,
	42, 24, ACTIVATION_TANH
}, vad_output{
	vad_output_bias,
	vad_output_weights,
	24, 1, ACTIVATION_SIGMOID
}, denoise_output{
	denoise_output_bias,
	denoise_output_weights,
	96, 22, ACTIVATION_SIGMOID
};

static constexpr GRULayer vad_gru{
	vad_gru_bias,
	vad_gru_weights,
	vad_gru_recurrent_weights,
	24, 24, ACTIVATION_RELU
}, noise_gru{
	noise_gru_bias,
	noise_gru_weights,
	noise_gru_recurrent_weights,
	90, 48, ACTIVATION_RELU
}, denoise_gru{
	denoise_gru_bias,
	denoise_gru_weights,
	denoise_gru_recurrent_weights,
	114, 96, ACTIVATION_RELU
};

static constexpr float tansig_table[]{
	0.000000f, 0.039979f, 0.079830f, 0.119427f, 0.158649f, 0.197375f, 0.235496f, 0.272905f, 0.309507f, 0.345214f, 0.379949f, 0.413644f, 0.446244f, 0.477700f, 0.507977f, 0.537050f, 0.564900f, 0.591519f, 0.616909f, 0.641077f, 0.664037f, 0.685809f, 0.706419f, 0.725897f, 0.744277f, 0.761594f, 0.777888f, 0.793199f, 0.807569f, 0.821040f, 0.833655f, 0.845456f, 0.856485f, 0.866784f, 0.876393f, 0.885352f, 0.893698f, 0.901468f, 0.908698f, 0.915420f, 0.921669f, 0.927473f, 0.932862f, 0.937863f, 0.942503f, 0.946806f, 0.950795f, 0.954492f, 0.957917f, 0.961090f, 0.964028f, 0.966747f, 0.969265f, 0.971594f, 0.973749f, 0.975743f, 0.977587f, 0.979293f, 0.980869f, 0.982327f, 0.983675f, 0.984921f, 0.986072f, 0.987136f, 0.988119f, 0.989027f, 0.989867f, 0.990642f, 0.991359f, 0.992020f, 0.992631f, 0.993196f, 0.993718f, 0.994199f, 0.994644f, 0.995055f, 0.995434f, 0.995784f, 0.996108f, 0.996407f, 0.996682f, 0.996937f, 0.997172f, 0.997389f, 0.997590f, 0.997775f, 0.997946f, 0.998104f, 0.998249f, 0.998384f, 0.998508f, 0.998623f, 0.998728f, 0.998826f, 0.998916f, 0.999000f, 0.999076f, 0.999147f, 0.999213f, 0.999273f, 0.999329f, 0.999381f, 0.999428f, 0.999472f, 0.999513f, 0.999550f, 0.999585f, 0.999617f, 0.999646f, 0.999673f, 0.999699f, 0.999722f, 0.999743f, 0.999763f, 0.999781f, 0.999798f, 0.999813f, 0.999828f, 0.999841f, 0.999853f, 0.999865f, 0.999875f, 0.999885f, 0.999893f, 0.999902f, 0.999909f, 0.999916f, 0.999923f, 0.999929f, 0.999934f, 0.999939f, 0.999944f, 0.999948f, 0.999952f, 0.999956f, 0.999959f, 0.999962f, 0.999965f, 0.999968f, 0.999970f, 0.999973f, 0.999975f, 0.999977f, 0.999978f, 0.999980f, 0.999982f, 0.999983f, 0.999984f, 0.999986f, 0.999987f, 0.999988f, 0.999989f, 0.999990f, 0.999990f, 0.999991f, 0.999992f, 0.999992f, 0.999993f, 0.999994f, 0.999994f, 0.999994f, 0.999995f, 0.999995f, 0.999996f, 0.999996f, 0.999996f, 0.999997f, 0.999997f, 0.999997f, 0.999997f, 0.999997f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f
};
inline static float tansig_approx(float x) {
	if (x >= 8)
		return 1;
	if (x <= -8)
		return -1;
	float sign;
	if (x < 0) {
		x = -x;
		sign = -1;
	}
	else {
		sign = 1;
	}
	const auto i = floor(.5f + 25 * x);
	x -= .04f * i;
	const auto y = tansig_table[static_cast<int>(i)];
	return sign * (y + x * (1 - y * y) * (1 - y * x));
}

inline static float sigmoid_approx(float x) {
	return .5f * (1 + tansig_approx(.5f * x));
}

inline static float relu(float x) {
	return x < 0 ? 0 : x;
}

static void compute_dense(const DenseLayer *layer, float *output, const float *input) {
	const auto M = layer->nb_inputs, N = layer->nb_neurons;
	int i;
	for (i = 0; i < N; ++i) {
		float sum{ static_cast<float>(layer->bias[i]) };
		for (int j = 0; j < M; ++j)
			sum += layer->input_weights[j * N + i] * input[j];
		output[i] = WEIGHTS_SCALE * sum;
	}
	switch (layer->activation)
	{
	case ACTIVATION_SIGMOID:
		for (i = 0; i < N; ++i)
			output[i] = sigmoid_approx(output[i]);
		break;
	case ACTIVATION_TANH:
		for (i = 0; i < N; ++i)
			output[i] = tansig_approx(output[i]);
		break;
	case ACTIVATION_RELU:
		for (i = 0; i < N; ++i)
			output[i] = relu(output[i]);
		break;
	}
}

static void compute_gru(const GRULayer *gru, float *state, const float *input) {
	const auto M = gru->nb_inputs, N = gru->nb_neurons, stride = 3 * N;
	int i, j;
	float z[MAX_NEURONS], r[MAX_NEURONS], h[MAX_NEURONS];
	for (i = 0; i < N; ++i) {
		float sum{ static_cast<float>(gru->bias[i]) };
		for (j = 0; j < M; ++j)
			sum += gru->input_wieghts[j * stride + i] * input[j];
		for (j = 0; j < N; ++j)
			sum += gru->recurrent_weights[j * stride + i] * state[j];
		z[i] = sigmoid_approx(WEIGHTS_SCALE * sum);
	}
	for (i = 0; i < N; ++i) {
		float sum{ static_cast<float>(gru->bias[N + i]) };
		for (j = 0; j < M; ++j)
			sum += gru->input_wieghts[N + j * stride + i] * input[j];
		for (j = 0; j < N; ++j)
			sum += gru->recurrent_weights[N + j * stride + i] * state[j];
		r[i] = sigmoid_approx(WEIGHTS_SCALE * sum);
	}
	for (i = 0; i < N; ++i) {
		float sum{ static_cast<float>(gru->bias[2 * N + i]) };
		for (j = 0; j < M; ++j)
			sum += gru->input_wieghts[2 * N + j * stride + i] * input[j];
		for (j = 0; j < N; ++j)
			sum += gru->recurrent_weights[2 * N + j * stride + i] * state[j] * r[j];
		sum = gru->activation == ACTIVATION_SIGMOID ? sigmoid_approx(WEIGHTS_SCALE * sum) :
			gru->activation == ACTIVATION_TANH ? tansig_approx(WEIGHTS_SCALE * sum) :
			gru->activation == ACTIVATION_RELU ? relu(WEIGHTS_SCALE * sum) :
			sum;
		h[i] = z[i] * state[i] + (1 - z[i]) * sum;
	}
	memcpy(state, h, sizeof(float) * N);
}

inline static void compute_rnn(RNNState &rnn, float *gains, float &vad, const float *input) {
	float dense_out[MAX_NEURONS], noise_input[MAX_NEURONS * 3], denoise_input[MAX_NEURONS * 3];
	compute_dense(&input_dense, dense_out, input);
	compute_gru(&vad_gru, rnn.vad_gru_state, dense_out);
	compute_dense(&vad_output, &vad, rnn.vad_gru_state);
	memcpy(noise_input, dense_out, sizeof(float) * INPUT_DENSE_SIZE);
	memcpy(noise_input + INPUT_DENSE_SIZE, rnn.vad_gru_state, sizeof(float) * VAD_GRU_SIZE);
	memcpy(noise_input + INPUT_DENSE_SIZE + VAD_GRU_SIZE, input, sizeof(float) * INPUT_SIZE);
	compute_gru(&noise_gru, rnn.noise_gru_state, noise_input);
	memcpy(denoise_input, rnn.vad_gru_state, sizeof(float) * VAD_GRU_SIZE);
	memcpy(denoise_input + VAD_GRU_SIZE, rnn.noise_gru_state, sizeof(float) * NOISE_GRU_SIZE);
	memcpy(denoise_input + VAD_GRU_SIZE + NOISE_GRU_SIZE, input, sizeof(float) * INPUT_SIZE);
	compute_gru(&denoise_gru, rnn.denoise_gru_state, denoise_input);
	compute_dense(&denoise_output, gains, rnn.denoise_gru_state);
}

static void interp_band_gain(float g[FREQ_SIZE], const float bandE[NB_BANDS]) {
	int k;
	for (int i{}; i < NB_BANDS - 1; ++i) {
		const auto band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
		for (int j{}; j < band_size; ++j) {
			float frac{ static_cast<float>(j) / band_size };
			g[k = (eband5ms[i] << FRAME_SIZE_SHIFT) + j] = (1 - frac) * bandE[i] + frac * bandE[i + 1];
		}
	}
	++k;
	memset(g + k, 0, sizeof(float) * (FREQ_SIZE - k));
}

inline static void pitch_filter(kiss_fft_cpx X[FREQ_SIZE], const kiss_fft_cpx P[WINDOW_SIZE],
	const float Ex[NB_BANDS], const float Ep[NB_BANDS], const float Exp[NB_BANDS], const float g[NB_BANDS]) {
	float r[NB_BANDS], rf[FREQ_SIZE], newE[NB_BANDS], norm[NB_BANDS], normf[FREQ_SIZE]{};
	int i;
	for (i = 0; i < NB_BANDS; ++i) {
		const auto _Exp = Exp[i], _Exp2 = _Exp * _Exp, _g = g[i], _g2 = _g * _g;
		r[i] = sqrt(max(0.f, min((_Exp > _g) ? 1 : _Exp2 * (1 - _g2) / (.001f + _g2 * (1 - _Exp2)), 1.f))) * sqrt(Ex[i] / (1e-8f + Ep[i]));
	}
	interp_band_gain(rf, r);
	for (i = 0; i < FREQ_SIZE; ++i)
		X[i] += rf[i] * P[i];
	compute_band_energy(newE, X);
	for (i = 0; i < NB_BANDS; ++i)
		norm[i] = sqrt(Ex[i] / (1e-8f + newE[i]));
	interp_band_gain(normf, norm);
	for (i = 0; i < FREQ_SIZE; ++i)
		X[i] *= normf[i];
}

inline static void inverse_transform(float out[WINDOW_SIZE], const kiss_fft_cpx in[FREQ_SIZE]) {
	kiss_fft_cpx x[WINDOW_SIZE], y[WINDOW_SIZE];
	int i;
	for (i = 0; i < FREQ_SIZE; ++i)
		x[i] = in[i];
	for (; i < WINDOW_SIZE; ++i)
		x[i] = conj(x[WINDOW_SIZE - i]);
	opus_fft_c(common.kfft, x, y);
	out[0] = WINDOW_SIZE * y->real();
	for (i = 1; i < WINDOW_SIZE; ++i)
		out[i] = WINDOW_SIZE * y[WINDOW_SIZE - i].real();
}

inline static void frame_synthesis(float synthesis_mem[FRAME_SIZE], float out[FRAME_SIZE], const kiss_fft_cpx y[FREQ_SIZE]) {
	static constexpr auto INT16_MAX_1 = 1 / 32767.f;
	float x[WINDOW_SIZE];
	inverse_transform(x, y);
	apply_window(x);
	for (int i{}; i < FRAME_SIZE; ++i)
		out[i] = (x[i] + synthesis_mem[i]) * INT16_MAX_1;
	memcpy(synthesis_mem, x + FRAME_SIZE, sizeof(float) * FRAME_SIZE);
}

struct RNNoise::State : DenoiseState { };

RNNoise::RNNoise() : st{ *new State{} } { }

RNNoise::~RNNoise() { delete &st; }

float RNNoise::transform(float out[480], const float in[480])
{
	kiss_fft_cpx X[FREQ_SIZE], P[WINDOW_SIZE];
	float x[FRAME_SIZE],
		Ex[NB_BANDS], Ep[NB_BANDS],
		Exp[NB_BANDS],
		features[NB_FEATURES],
		g[NB_BANDS],
		gf[FREQ_SIZE]{ 1 },
		vad_prob{};
	static constexpr float a_hp[2]{ -1.99599f, .996f }, b_hp[2]{ -2, 1 };
	biquad(x, st.mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
	if (!compute_frame_features(st, X, P, Ex, Ep, Exp, features, x)) {
		compute_rnn(st.rnn, g, vad_prob, features);
		pitch_filter(X, P, Ex, Ep, Exp, g);
		int i;
		for (i = 0; i < NB_BANDS; ++i)
			st.lastg[i] = g[i] = max(g[i], .6f * st.lastg[i]);
		interp_band_gain(gf, g);
		for (i = 0; i < FREQ_SIZE; ++i)
			X[i] *= gf[i];
	}
	frame_synthesis(st.synthesis_mem, out, X);
	return vad_prob;
}
