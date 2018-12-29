package goceptron

import (
	"gonum.org/v1/gonum/mat"
)

type NetInputParams struct {
	Input   *mat.Dense
	Weights *mat.Dense
	Bias    float64
}

func NetInput(params NetInputParams) mat.Dense {

	var output mat.Dense
	output.Mul(params.Input, params.Weights.T())

	output.Apply(func(i, j int, v float64) float64 {
		return v + params.Bias
	}, &output)

	return output
}
