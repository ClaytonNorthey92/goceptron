package goceptron

import (
	"math/rand"
)

func GenerateRandomWeights(weightCount int, stddev float64) ([]float64, float64) {
	weights := make([]float64, weightCount)

	for i := 0; i < weightCount; i++ {
		weights[i] = rand.NormFloat64() * stddev
	}

	return weights, rand.NormFloat64() * stddev
}
