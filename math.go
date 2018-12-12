package goceptron

import (
	"gonum.org/v1/gonum/mat"
)

type NetInputParams struct {
	Input            []float64
	Weights          []float64
	InputRowCount    int
	InputColumnCount int
}

func NetInput(params NetInputParams) mat.Dense {

	inputMatrix := mat.NewDense(params.InputRowCount, params.InputColumnCount, params.Input)
	weightMatrix := mat.NewDense(1, len(params.Weights), params.Weights).T()
	var output mat.Dense
	output.Mul(inputMatrix, weightMatrix)
	return output
}
