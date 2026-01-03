// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

// ArchitectureType defines the model architecture.
type ArchitectureType int

const (
	// CBOW (Continuous Bag-of-Words) predicts a word from its context.
	CBOW ArchitectureType = iota
	// SkipGram predicts the context from a word.
	SkipGram
)

// LossFunctionType defines the loss function for training.
type LossFunctionType int

const (
	// NegativeSampling is an efficient training method.
	NegativeSampling LossFunctionType = iota
	// HierarchicalSoftmax is an alternative method that works well for rare words.
	HierarchicalSoftmax
)

// TrainSettings holds all parameters for model training.
type TrainSettings struct {
	// Dimensionality of the word vectors.
	VectorSize int
	// Initial learning rate.
	LearningRate float32
	// Model architecture (CBOW or SkipGram).
	Architecture ArchitectureType
	// Loss function (NegativeSampling or HierarchicalSoftmax).
	LossFunction LossFunctionType
	// Size of the context window.
	WindowSize int
	// Threshold for discarding rare words.
	MinCount int
	// Number of negative examples for Negative Sampling.
	NegativeSamples int
	// Threshold for subsampling frequent words.
	SubsamplingThreshold float32
	// Number of threads for training.
	NumThreads int
	// Number of training epochs.
	NumEpochs int
	// Flag for verbose output.
	Verbose bool
}
