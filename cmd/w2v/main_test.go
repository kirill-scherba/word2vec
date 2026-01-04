package main

import (
	"bytes"
	"testing"

	"github.com/kirill-scherba/word2vec/internal/libw2v"
)

// Mock Vocabulary for testing
type mockVocabulary struct {
	words []libw2v.VocabWord
	stems map[string]string
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

func (m *mockVocabulary) StemMap() map[string]string {
	return m.stems
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
	err := libw2v.SaveModel(mockMod, mockVocab, false, &buffer)
	if err != nil {
		t.Fatalf("saveModel failed: %v", err)
	}

	// 3. Load the model from the buffer
	loadedWords, loadedVectors, lemmatized, stemMap, err := libw2v.LoadModel(&buffer)
	if err != nil {
		t.Fatalf("loadModel failed: %v", err)
	}

	// 4. Compare the results
	if len(loadedWords) != len(originalWords) {
		t.Fatalf("expected %d words, got %d", len(originalWords), len(loadedWords))
	}

	if lemmatized {
		t.Error("expected lemmatized flag to be false, but got true")
	}

	if len(stemMap) != 0 {
		t.Errorf("expected empty stemMap, but got %d entries", len(stemMap))
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

func TestSaveAndLoadModelWithStemming(t *testing.T) {
	// 1. Create dummy data for a stemmed model
	originalWords := []libw2v.VocabWord{
		{Word: "корол", Count: 100}, // stem of "король"
		{Word: "женщин", Count: 90}, // stem of "женщина"
	}
	originalVectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}
	originalStemMap := map[string]string{
		"корол":  "король",
		"женщин": "женщина",
	}

	mockVocab := &mockVocabulary{words: originalWords, stems: originalStemMap}
	mockMod := &mockModel{vectors: originalVectors}

	// 2. Save the model with lemmatize=true
	var buffer bytes.Buffer
	err := libw2v.SaveModel(mockMod, mockVocab, true, &buffer)
	if err != nil {
		t.Fatalf("saveModel with stemming failed: %v", err)
	}

	// 3. Load the model from the buffer
	loadedWords, _, lemmatized, loadedStemMap, err := libw2v.LoadModel(&buffer)
	if err != nil {
		t.Fatalf("loadModel with stemming failed: %v", err)
	}

	// 4. Compare the results
	if !lemmatized {
		t.Error("expected lemmatized flag to be true, but got false")
	}

	if len(loadedStemMap) != len(originalStemMap) {
		t.Fatalf("expected stemMap of size %d, got %d", len(originalStemMap), len(loadedStemMap))
	}

	for stem, original := range originalStemMap {
		if loadedStemMap[stem] != original {
			t.Errorf("stemMap mismatch for stem '%s': expected '%s', got '%s'", stem, original, loadedStemMap[stem])
		}
	}

	// Also check words and vectors
	if loadedWords[0] != "корол" {
		t.Errorf("word mismatch: expected 'корол', got '%s'", loadedWords[0])
	}
}
