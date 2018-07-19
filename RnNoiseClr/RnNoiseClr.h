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

		float Transform(array<short> ^out, array<const short> ^in);
	};
}