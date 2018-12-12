package goceptron

import (
	"errors"
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Perceptron interface {
	Predict(data []float64) (int, error)
	Fit(trainingSet [][]float64, resultSet []int) error
}

type PerceptronBasic struct {
	LearningRate float64
	Epochs       int
	RandomSeed   int64
	weights      []float64
	bias         float64
}

type PerceptronAdalineGD struct {
	PerceptronBasic
}

func (p *PerceptronBasic) Predict(data []float64) (int, error) {
	if len(p.weights) != len(data) {
		return 0, errors.New(fmt.Sprintf("input data and weights do not have the same length: %d != %d", len(data), len(p.weights)))
	}

	total := NetInput(NetInputParams{
		Weights:          p.weights,
		Input:            data,
		InputColumnCount: len(data),
		InputRowCount:    1,
	})

	if mat.Trace(&total)+p.bias >= 0 {
		return 1, nil
	} else {
		return -1, nil
	}
}

func (p *PerceptronBasic) Fit(trainingSet [][]float64, resultSet []int) error {
	rand.Seed(p.RandomSeed)
	weightCount := len(trainingSet[0])
	p.weights, p.bias = GenerateRandomWeights(weightCount)

	if len(trainingSet) != len(resultSet) {
		return errors.New(fmt.Sprintf("training set length does not match result set row length: %d != %d", len(trainingSet), len(resultSet)))
	}

	for i := 0; i < p.Epochs; i++ {
		for i, row := range trainingSet {
			prediction, err := p.Predict(row)
			if err != nil {
				return err
			}
			update := p.LearningRate * float64(resultSet[i]-prediction)
			p.updateWeights(update, row)
		}
	}

	return nil
}

func (p *PerceptronBasic) updateWeights(update float64, input []float64) {
	p.bias += update
	for i, _ := range p.weights {
		p.weights[i] += update * input[i]
	}
}

func (p *PerceptronAdalineGD) Fit(trainingSet [][]float64, resultSet []int) error {
	rand.Seed(p.RandomSeed)
	weightCount := len(trainingSet[0])
	p.weights, p.bias = GenerateRandomWeights(weightCount)

	for i := 0; i < p.Epochs; i++ {
	}

	return nil
}
