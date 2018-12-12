package goceptron

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func Test_NetInput(t *testing.T) {
	input := []float64{8.0, 9.0, 8.0}
	weights := []float64{5.0, 1.0, 9.0}
	expectedOutput := (8.0 * 5.0) + (9.0 * 1.0) + (8.0 * 9.0)
	output := NetInput(NetInputParams{
		Input:            input,
		Weights:          weights,
		InputRowCount:    1,
		InputColumnCount: 3,
	})

	outputSum := mat.Trace(&output)

	if outputSum != expectedOutput {
		t.Errorf("output %f is not equal to %f", outputSum, expectedOutput)
	}
}
