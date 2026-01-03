// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import (
	"math/rand"
	"sync"
)

// Model holds the neural network weights (embeddings).
type Model struct {
	// Input to hidden layer weights. This is the main word embedding matrix.
	syn0 [][]float32

	// Hidden to output layer weights. Used for Hierarchical Softmax.
	syn1 [][]float32

	// Hidden to output layer weights. Used for Negative Sampling.
	syn1neg [][]float32

	// Mutex to protect concurrent access to weights.
	mu sync.Mutex
}

// NewModel creates and initializes a new word2vec model.
func NewModel(vocab *Vocabulary, settings TrainSettings) *Model {
	vocabSize := vocab.Len()
	vectorSize := settings.VectorSize

	// Allocate memory for the network weights
	syn0 := make([][]float32, vocabSize)
	for i := range syn0 {
		syn0[i] = make([]float32, vectorSize)
	}

	// Initialize syn0 with random values, as in the original C code.
	for i := range vocabSize {
		for j := 0; j < vectorSize; j++ {
			// Random values between -0.5/vectorSize and +0.5/vectorSize
			syn0[i][j] = (rand.Float32() - 0.5) / float32(vectorSize)
		}
	}

	var syn1, syn1neg [][]float32

	if settings.LossFunction == HierarchicalSoftmax {
		syn1 = make([][]float32, vocabSize)
		for i := range syn1 {
			syn1[i] = make([]float32, vectorSize) // Initialized to zeros
		}
	}

	if settings.NegativeSamples > 0 {
		syn1neg = make([][]float32, vocabSize)
		for i := range syn1neg {
			syn1neg[i] = make([]float32, vectorSize) // Initialized to zeros
		}
	}

	return &Model{
		syn0:    syn0,
		syn1:    syn1,
		syn1neg: syn1neg,
	}
}

// ExportSyn0 returns the input-to-hidden layer weights (the word embeddings).
func (m *Model) ExportSyn0() [][]float32 {
	return m.syn0
}
