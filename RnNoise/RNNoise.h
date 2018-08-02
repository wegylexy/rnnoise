#pragma once
#include <memory>

class RNNoise
{
	struct State;
	State &st;

public:
	RNNoise();
	~RNNoise();

	float transform(float out[480], const float in[480]);
};

