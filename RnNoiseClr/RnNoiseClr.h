#pragma once

#include "../RnNoise/RNNoise.h"

namespace RnNoiseClr {
	public ref class RNNoiseCLR
	{
		RNNoise *native;

	public:
		literal int FRAME_SIZE = 480;

		RNNoiseCLR();
		!RNNoiseCLR();
		~RNNoiseCLR();

		inline float Transform(array<short> ^out, array<const short> ^in) { return Transform(out, 0, in, 0); }
		float Transform(array<short> ^out, int out_offset, array<const short> ^in, int in_offset);
	};
}