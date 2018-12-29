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
	Stddev       float64
	weights      *mat.Dense
	bias         float64
}

type PerceptronAdalineGD struct {
	PerceptronBasic
}

func (p *PerceptronBasic) Predict(data []float64) (int, error) {
	inputData := mat.NewDense(1, len(data), data)
	total := NetInput(NetInputParams{
		Weights: p.weights,
		Input:   inputData,
		Bias:    p.bias,
	})

	if mat.Sum(&total) >= 0 {
		return 1, nil
	} else {
		return -1, nil
	}
}

func (p *PerceptronBasic) Fit(trainingSet [][]float64, resultSet []int) error {
	rand.Seed(p.RandomSeed)
	weightCount := len(trainingSet[0])
	weights, bias := GenerateRandomWeights(weightCount, p.Stddev)

	p.weights = mat.NewDense(1, weightCount, weights)
	p.bias = bias

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
	for i, _ := range p.weights.RawRowView(0) {
		newValue := p.weights.At(0, i) + update*input[i]
		p.weights.Set(0, i, newValue)
	}
}

func (p *PerceptronAdalineGD) Fit(trainingSet [][]float64, resultSet []int) error {
	resultSetConverted := make([]float64, len(resultSet))
	for i, v := range resultSet {
		resultSetConverted[i] = float64(v)
	}

	resultsSetDense := mat.NewDense(len(resultSetConverted), 1, resultSetConverted)

	rand.Seed(p.RandomSeed)
	weightCount := len(trainingSet[0])

	weights, bias := GenerateRandomWeights(weightCount, p.Stddev)

	p.weights = mat.NewDense(1, weightCount, weights)
	p.bias = bias

	flattenedData := Flatten(trainingSet)

	inputData := mat.NewDense(len(trainingSet), len(trainingSet[0]), flattenedData)

	for i := 0; i < p.Epochs; i++ {
		netInput := NetInput(NetInputParams{
			Weights: p.weights,
			Input:   inputData,
			Bias:    p.bias,
		})

		rows, cols := netInput.Dims()

		errors := mat.NewDense(rows, cols, nil)

		errors.Sub(resultsSetDense, &netInput)

		weightUpdate := mat.NewDense(1, weightCount, nil)
		weightUpdate.Mul(errors.T(), inputData)

		weightUpdate.Scale(p.LearningRate, weightUpdate)

		p.weights.Apply(func(i, j int, v float64) float64 {
			return v + weightUpdate.At(i, j)
		}, p.weights)

		errorSum := mat.Sum(errors)
		p.bias += p.LearningRate * errorSum

	}

	return nil
}
