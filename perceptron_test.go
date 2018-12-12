package goceptron

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"testing"

	shuffle "github.com/shogo82148/go-shuffle"
)

func TestPerceptron(t *testing.T) {
	perceptron := PerceptronBasic{
		LearningRate: 0.1,
		Epochs:       50,
		RandomSeed:   1,
	}

	trainingSet := [][]float64{
		{0.9, 0.8},
		{0.2, 0.7},
	}

	resultsSet := []int{1, -1}

	perceptron.Fit(trainingSet, resultsSet)
}

type dataset struct {
	Data    [][]float64
	Results []int
}

func (d *dataset) Len() int {
	return len(d.Data)
}

func (d *dataset) Swap(i, j int) {
	tmp := d.Data[i]
	tmpR := d.Results[i]

	d.Data[i] = d.Data[j]
	d.Results[i] = d.Results[j]

	d.Data[j] = tmp
	d.Results[j] = tmpR
}

func TestPerceptronBinaryClassifier(t *testing.T) {
	file, err := os.Open("./datasets/iris.csv")
	if err != nil {
		t.Error(err.Error())
	}
	defer file.Close()

	rows, err := csv.NewReader(file).ReadAll()
	if err != nil {
		t.Error(err.Error())
	}

	data := make([][]float64, 100)
	results := make([]int, 100)

	usedRows := rows[:100]

	for i, row := range usedRows {
		var featuresRaw []string = row[:4]
		features := make([]float64, 4)
		for i, f := range featuresRaw {
			features[i], err = strconv.ParseFloat(f, 64)
			if err != nil {
				t.Error(err.Error())
			}
		}
		data[i] = features
		if row[4] == "Iris-setosa" {
			results[i] = 1
		} else {
			results[i] = -1
		}
	}

	d := dataset{
		Data:    data,
		Results: results,
	}

	shuffle.Shuffle(&d)

	trainingData := d.Data[:66]
	trainingResults := d.Results[:66]

	testData := d.Data[66:100]
	testResults := d.Results[66:100]

	perceptron := PerceptronBasic{
		LearningRate: 0.15,
		Epochs:       50,
		RandomSeed:   1,
	}

	perceptron.Fit(trainingData, trainingResults)

	for i, v := range testData {
		prediction, err := perceptron.Predict(v)
		if err != nil {
			t.Error(err.Error())
		}

		if prediction != testResults[i] {
			fmt.Println(prediction, testResults[i], testData[i])
			t.Error("incorrect prediction")
		}
	}

}

func TestPerceptronBinaryClassifierSonar(t *testing.T) {
	file, err := os.Open("./datasets/sonar.csv")
	if err != nil {
		t.Error(err.Error())
	}
	defer file.Close()

	rows, err := csv.NewReader(file).ReadAll()
	if err != nil {
		t.Error(err.Error())
	}

	data := make([][]float64, 208)
	results := make([]int, 208)

	usedRows := rows[:208]

	for i, row := range usedRows {
		var featuresRaw []string = row[:60]
		features := make([]float64, 60)
		for i, f := range featuresRaw {
			features[i], err = strconv.ParseFloat(f, 64)
			if err != nil {
				t.Error(err.Error())
			}
		}
		data[i] = features
		if row[60] == "M" {
			results[i] = 1
		} else {
			results[i] = -1
		}
	}

	d := dataset{
		Data:    data,
		Results: results,
	}

	shuffler := shuffle.New(rand.NewSource(1))

	shuffler.Shuffle(&d)

	trainingData := d.Data[:140]
	trainingResults := d.Results[:140]

	testData := d.Data[140:208]
	testResults := d.Results[140:208]

	perceptron := PerceptronBasic{
		LearningRate: 0.1,
		Epochs:       50000,
		RandomSeed:   1,
	}

	perceptron.Fit(trainingData, trainingResults)

	total := len(testData)
	var errorCount int

	for i, v := range testData {
		prediction, err := perceptron.Predict(v)
		if err != nil {
			t.Error(err.Error())
		}

		if prediction != testResults[i] {
			errorCount++
		}
	}

	errorRate := float64(errorCount) / float64(total)
	if errorRate > 0.12 {
		fmt.Sprintf("Correct Predictions: %d -- Incorrect Predictions: %d", total-errorCount, errorCount)
		t.Errorf("Sonar model is not accurate enough, error rate: %f", errorRate)
	}

}
