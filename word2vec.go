// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The word2vec package provides a native Go implementation of the word2vec algorithm.
// It allows loading pre-trained models and using them to find word embeddings
// and semantic similarities.
package word2vec

import (
	"errors"
	"fmt"
	"io"
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
	lemmatized bool
	stemMap    map[string]string
}

// Load reads a pre-trained model from the specified file path.
func Load(modelFile string) (*Model, error) {

	// Open the model file
	file, err := os.Open(modelFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Load the model using the reader
	return LoadReader(file)
}

// LoadReader reads a pre-trained model from the specified reader.
func LoadReader(r io.Reader) (*Model, error) {

	// Load model
	words, vectors, lemmatized, stemMap, err := libw2v.LoadModel(r)
	if err != nil {
		return nil, fmt.Errorf("failed to load model data: %w", err)
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

	// Return the loaded model
	return &Model{
		words:      words,
		vectors:    vectors,
		wordMap:    wordMap,
		vectorSize: len(vectors[0]),
		lemmatized: lemmatized,
		stemMap:    stemMap,
	}, nil
}

// Size returns the size of the word vectors in the model.
func (m *Model) Size() int {
	return m.vectorSize
}

// Lemmatized returns true if the model was trained with lemmatization/stemming.
func (m *Model) Lemmatized() bool {
	return m.lemmatized
}

// ProcessWord cleans and optionally stems a word based on the model's properties.
func (m *Model) ProcessWord(word string) string {
	return m.processWord(word)
}

// processWord cleans and optionally stems a word based on the model's properties.
func (m *Model) processWord(word string) string {
	cleaned := libw2v.CleanWord(word)
	if m.lemmatized {
		return libw2v.Stem(cleaned)
	}
	return cleaned
}

// VectorOf calculates the embedding vector for a single input word.
func (m *Model) VectorOf(word string, vector []float32) error {
	processedWord := m.ProcessWord(word)

	idx, exists := m.wordMap[processedWord]
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
		cleaned := m.ProcessWord(word)
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

// Similarity calculates the cosine similarity between two documents.
// A score of 1 means the texts are very similar, and a score of -1 means they are very dissimilar.
func (m *Model) Similarity(doc1, doc2 string) (float32, error) {
	vec1 := make([]float32, m.vectorSize)
	if err := m.Embedding(doc1, vec1); err != nil {
		return 0, fmt.Errorf("failed to get embedding for doc1: %w", err)
	}

	vec2 := make([]float32, m.vectorSize)
	if err := m.Embedding(doc2, vec2); err != nil {
		return 0, fmt.Errorf("failed to get embedding for doc2: %w", err)
	}

	// The vectors from Embedding are averaged, so we need to re-normalize them
	// before calculating the dot product to get the cosine similarity.
	libw2v.Normalize(vec1)
	libw2v.Normalize(vec2)

	return libw2v.Dot(vec1, vec2), nil
}

// Nearest represents a word and its similarity score.
type Nearest struct {
	Word     string
	Distance float32
}

// Lookup finds the most similar words in the model for a given query word.
func (m *Model) Lookup(query string, seq []Nearest) error {
	cleanedQuery := m.ProcessWord(query)

	queryIdx, exists := m.wordMap[cleanedQuery]
	if !exists {
		return errors.New("query word not in vocabulary")
	}
	queryVector := m.vectors[queryIdx]

	distances := make([]Nearest, 0, len(m.words))
	for i, w := range m.words {
		// Exclude the query word itself from the results
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
		res := distances[i]
		// If we have a stem map, replace the stem with the original word.
		if m.lemmatized && m.stemMap != nil {
			res.Word = m.stemMap[res.Word]
		}
		seq[i] = res
	}

	return nil
}

// NearestToVector finds the most similar words in the model for a given vector.
// The `exclude` map can be used to ignore certain words in the result.
func (m *Model) NearestToVector(vector []float32, seq []Nearest, exclude map[string]bool) error {
	if len(vector) != m.vectorSize {
		return errors.New("vector size does not match model's vector size")
	}

	if exclude == nil {
		exclude = make(map[string]bool)
	}

	// Normalize the input vector to ensure cosine similarity calculation is correct.
	libw2v.Normalize(vector)

	distances := make([]Nearest, 0, len(m.words))
	for i, w := range m.words {
		// Skip excluded words
		if exclude[w] {
			continue
		}

		dist := libw2v.Dot(vector, m.vectors[i])
		distances = append(distances, Nearest{Word: w, Distance: dist})
	}

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].Distance > distances[j].Distance
	})

	for i := 0; i < len(seq) && i < len(distances); i++ {
		res := distances[i]
		// If we have a stem map, replace the stem with the original word.
		if m.lemmatized && m.stemMap != nil {
			res.Word = m.stemMap[res.Word]
		}
		seq[i] = res
	}

	return nil
}
