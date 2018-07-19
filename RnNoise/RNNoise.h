#pragma once
#include <memory>

class RNNoise
{
	struct State;
	std::unique_ptr<State> st;

public:
	RNNoise();
	~RNNoise();

	float transform(short out[480], const short in[480]);
};

