package goceptron

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

func Flatten(input [][]float64) []float64 {
	flattenedData := make([]float64, len(input)*len(input[0]))

	count := 0
	for i, _ := range input {
		for j, _ := range input[i] {
			flattenedData[count] = input[i][j]
			count++
		}
	}

	return flattenedData
}

func Normalize(data [][]float64) {
	flatData := mat.NewDense(len(data[0]), len(data), Flatten(data))

	for i := 0; i < len(data[0]); i++ {
		features := RawColView(flatData, i)
		mean, stddev := stat.MeanStdDev(features, nil)
		for j := 0; j < len(data); j++ {
			data[j][i] = (data[j][i] - mean) / stddev
		}
	}
}

func RawColView(d *mat.Dense, col int) []float64 {
	rows, _ := d.Dims()
	results := mat.NewVecDense(rows, nil)
	results.CopyVec(d.ColView(col))
	return results.RawVector().Data
}
