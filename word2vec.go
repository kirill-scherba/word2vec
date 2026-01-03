// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The word2vec package provides a native Go implementation of the word2vec algorithm.
// It allows loading pre-trained models and using them to find word embeddings
// and semantic similarities.
package word2vec

import (
	"errors"
	"os"
	"sort"
	"strings"

	"github.com/kirill-scherba/word2vec/internal/libw2v"
)

// Model holds the loaded word vector model.
type Model struct {
	words      []string
	vectors    [][]float32
	wordMap    map[string]int // For fast lookups
	vectorSize int
}

// Load reads a pre-trained model from the specified file path.
func Load(modelFile string) (*Model, error) {
	file, err := os.Open(modelFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	words, vectors, err := libw2v.LoadModel(file)
	if err != nil {
		return nil, err
	}

	if len(vectors) == 0 {
		return nil, errors.New("model is empty")
	}

	// Create a word map for fast lookups
	wordMap := make(map[string]int, len(words))
	for i, w := range words {
		wordMap[w] = i
	}

	// Normalize all vectors for efficient cosine similarity calculation.
	for i := range vectors {
		libw2v.Normalize(vectors[i])
	}

	return &Model{
		words:      words,
		vectors:    vectors,
		wordMap:    wordMap,
		vectorSize: len(vectors[0]),
	}, nil
}

// Size returns the size of the word vectors in the model.
func (m *Model) Size() int {
	return m.vectorSize
}

// VectorOf calculates the embedding vector for a single input word.
func (m *Model) VectorOf(word string, vector []float32) error {
	idx, exists := m.wordMap[word]
	if !exists {
		return errors.New("word not in vocabulary")
	}
	copy(vector, m.vectors[idx])
	return nil
}

// Embedding calculates the embedding for a document (a bag of words)
// by averaging the vectors of the words in it.
func (m *Model) Embedding(doc string, vector []float32) error {
	// Clear the output vector
	for i := range vector {
		vector[i] = 0
	}

	words := strings.Fields(doc)
	count := 0
	for _, word := range words {
		cleaned := libw2v.CleanWord(word)
		if idx, exists := m.wordMap[cleaned]; exists {
			for i := 0; i < m.vectorSize; i++ {
				vector[i] += m.vectors[idx][i]
			}
			count++
		}
	}

	if count == 0 {
		return errors.New("no known words in the document")
	}

	// Average the vectors
	for i := range vector {
		vector[i] /= float32(count)
	}
	return nil
}

// Nearest represents a word and its similarity score.
type Nearest struct {
	Word     string
	Distance float32
}

// Lookup finds the most similar words in the model for a given query word.
func (m *Model) Lookup(query string, seq []Nearest) error {
	// Clean the query word to match the vocabulary format
	cleanedQuery := libw2v.CleanWord(query)

	queryIdx, exists := m.wordMap[cleanedQuery]
	if !exists {
		return errors.New("query word not in vocabulary")
	}
	queryVector := m.vectors[queryIdx]

	distances := make([]Nearest, 0, len(m.words))
	for i, w := range m.words {
		if i == queryIdx {
			continue
		}
		dist := libw2v.Dot(queryVector, m.vectors[i])
		distances = append(distances, Nearest{Word: w, Distance: dist})
	}

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].Distance > distances[j].Distance
	})

	for i := 0; i < len(seq) && i < len(distances); i++ {
		seq[i] = distances[i]
	}

	return nil
}
