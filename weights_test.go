package goceptron

import (
	"fmt"
	"testing"
)

func TestGenerateWeights(t *testing.T) {
	weights, _ := GenerateRandomWeights(6, 0.01)
	if len(weights) != 6 {
		t.Error(fmt.Sprintf("incorrect number of weights: %d", len(weights)))
	}
}
