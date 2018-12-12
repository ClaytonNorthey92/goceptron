package goceptron

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func Test_NetInput(t *testing.T) {
	input := mat.NewDense(1, 3, []float64{8.0, 9.0, 8.0})
	weights := mat.NewDense(1, 3, []float64{5.0, 1.0, 9.0})
	expectedOutput := (8.0 * 5.0) + (9.0 * 1.0) + (8.0 * 9.0)
	output := NetInput(NetInputParams{
		Input:   input,
		Weights: weights,
	})

	outputSum := mat.Sum(&output)

	if outputSum != expectedOutput {
		t.Errorf("output %f is not equal to %f", outputSum, expectedOutput)
	}
}

func Test_NetInputWithBias(t *testing.T) {
	input := mat.NewDense(1, 3, []float64{8.0, 9.0, 8.0})
	weights := mat.NewDense(1, 3, []float64{5.0, 1.0, 9.0})
	bias := 8.0
	expectedOutput := (8.0 * 5.0) + (9.0 * 1.0) + (8.0 * 9.0) + 8.0
	output := NetInput(NetInputParams{
		Input:   input,
		Weights: weights,
		Bias:    bias,
	})

	outputSum := mat.Trace(&output)

	if outputSum != expectedOutput {
		t.Errorf("output %f is not equal to %f", outputSum, expectedOutput)
	}
}
