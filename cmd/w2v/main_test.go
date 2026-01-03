package main

import (
	"bytes"
	"testing"

	"github.com/kirill-scherba/word2vec/internal/libw2v"
)

// Mock Vocabulary for testing
type mockVocabulary struct {
	words []libw2v.VocabWord
} // Mock Model for testing
type mockModel struct {
	vectors [][]float32
}

func (m *mockVocabulary) Words() []libw2v.VocabWord {
	return m.words
}

func (m *mockVocabulary) Len() int {
	return len(m.words)
}

func (m *mockModel) ExportSyn0() [][]float32 {
	return m.vectors
}

func TestSaveAndLoadModel(t *testing.T) {
	// 1. Create dummy data
	originalWords := []libw2v.VocabWord{
		{Word: "king", Count: 100},
		{Word: "queen", Count: 90},
		{Word: "man", Count: 80},
	}
	originalVectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
		{0.7, 0.8, 0.9},
	}

	mockVocab := &mockVocabulary{words: originalWords}
	mockMod := &mockModel{vectors: originalVectors}

	// 2. Save the model to an in-memory buffer
	var buffer bytes.Buffer
	err := libw2v.SaveModel(mockMod, mockVocab, &buffer)
	if err != nil {
		t.Fatalf("saveModel failed: %v", err)
	}

	// 3. Load the model from the buffer
	loadedWords, loadedVectors, err := libw2v.LoadModel(&buffer)
	if err != nil {
		t.Fatalf("loadModel failed: %v", err)
	}

	// 4. Compare the results
	if len(loadedWords) != len(originalWords) {
		t.Fatalf("expected %d words, got %d", len(originalWords), len(loadedWords))
	}

	for i := range originalWords {
		if loadedWords[i] != originalWords[i].Word {
			t.Errorf("word mismatch at index %d: expected %s, got %s", i, originalWords[i].Word, loadedWords[i])
		}
		for j := range originalVectors[i] {
			if loadedVectors[i][j] != originalVectors[i][j] {
				t.Errorf("vector mismatch at word %s, index %d: expected %f, got %f", loadedWords[i], j, originalVectors[i][j], loadedVectors[i][j])
			}
		}
	}
}
