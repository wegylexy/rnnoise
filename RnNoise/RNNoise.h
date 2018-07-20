#pragma once
#include <memory>

class RNNoise
{
	struct State;
	State &st;

public:
	RNNoise();
	~RNNoise();

	float transform(short out[480], const short in[480]);
};

