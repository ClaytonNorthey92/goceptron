package goceptron

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestRawColView(t *testing.T) {
	// 4 5
	// 9 8
	d := mat.NewDense(2, 2, []float64{4.0, 5.0, 9.0, 8.0})

	colView := RawColView(d, 1)
	if !reflect.DeepEqual(colView, []float64{5.0, 8.0}) {
		t.Error("incorrect raw column view")
	}
}
