#include "RnNoiseClr.h"

using namespace System;
using namespace RnNoiseClr;

RNNoiseCLR::RNNoiseCLR() : native{ new RNNoise{} } { }

RNNoiseCLR::!RNNoiseCLR() {
	delete native;
	native = nullptr;
}

RNNoiseCLR::~RNNoiseCLR() { this->!RNNoiseCLR(); }

float RNNoiseCLR::Transform(array<short> ^out, int out_offset, array<const short> ^in, int in_offset)
{
	if (!out)
		throw gcnew ArgumentNullException{ "out" };
	if (!in)
		throw gcnew ArgumentNullException{ "in" };
	if (out->Length != in->Length || out->Length - out_offset < FRAME_SIZE || in->Length - in_offset < FRAME_SIZE)
		throw gcnew ArgumentOutOfRangeException{ "Frame size must be multiple 480" };
	pin_ptr<short> _out{ &out[out_offset] };
	pin_ptr<const short> _in{ &in[in_offset] };
	return native->transform(_out, _in);
}