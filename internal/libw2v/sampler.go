// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import (
	"math"
	"math/rand"
)

const unigramTableSize = 1e8

// Sampler holds the structures needed for negative sampling and subsampling.
type Sampler struct {
	unigramTable []int
}

// NewSampler creates a sampler and initializes the unigram table for negative sampling.
func NewSampler(vocab *Vocabulary) *Sampler {
	table := make([]int, unigramTableSize)
	if vocab.Len() == 0 {
		return &Sampler{unigramTable: table}
	}

	// The power to which word frequencies are raised. 0.75 is the value from the original paper.
	const power = 0.75
	trainWordsPow := 0.0

	for _, word := range vocab.Words() {
		trainWordsPow += math.Pow(float64(word.Count), power)
	}

	i := 0
	d1 := math.Pow(float64(vocab.Words()[i].Count), power) / trainWordsPow
	for a := 0; a < unigramTableSize; a++ {
		table[a] = i
		if float64(a)/unigramTableSize > d1 {
			i++
			if i >= vocab.Len() { // Ensure we don't go out of bounds
				i = vocab.Len() - 1
			}
			d1 += math.Pow(float64(vocab.Words()[i].Count), power) / trainWordsPow
		}
	}

	return &Sampler{unigramTable: table}
}

// ShouldSubsample determines whether a word should be discarded based on its frequency.
// It returns true if the word should be kept, false if it should be discarded.
func (s *Sampler) ShouldSubsample(word *VocabWord, totalWords int64, threshold float32) bool {
	if threshold <= 0 {
		return true
	}
	wordFreq := float64(word.Count) / float64(totalWords)
	// This is the discard probability from the original word2vec paper.
	discardProbability := 1.0 - math.Sqrt(float64(threshold)/wordFreq)
	return rand.Float64() >= discardProbability
}
