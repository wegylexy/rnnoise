#pragma once
#include <memory>

class RNNoise
{
	struct State;
	std::unique_ptr<State> st;

public:
	RNNoise();
	~RNNoise();

	float transform(const short in[480], short out[480]);
};

