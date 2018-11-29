package perceptron

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"testing"

	"github.com/shogo82148/go-shuffle"
)

func TestPerceptron(t *testing.T) {
	perceptron := Perceptron{
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
	Data    [100][]float64
	Results [100]int
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

func TestPerceptronOneVsMany(t *testing.T) {
	file, err := os.Open("./datasets/iris.csv")
	if err != nil {
		t.Error(err.Error())
	}
	defer file.Close()

	rows, err := csv.NewReader(file).ReadAll()
	if err != nil {
		t.Error(err.Error())
	}

	var data [100][]float64
	var results [100]int

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

	perceptron := Perceptron{
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
