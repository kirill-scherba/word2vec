// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"fmt"
	"log"

	"github.com/kirill-scherba/word2vec"
	"github.com/spf13/cobra"
)

var (
	analogyModelFile string
	analogyWords     []string
	analogyTopN      int
)

var analogyCmd = &cobra.Command{
	Use:   "analogy",
	Short: "Perform word analogy tasks like 'king - man + woman = queen'",
	Long: `Performs a word analogy task.
It takes three words (word1, word2, word3) and computes word1 - word2 + word3.
Example: ./w2v analogy -m vectors.bin --words король,мужчина,женщина`,
	Run: runAnalogy,
}

func init() {
	rootCmd.AddCommand(analogyCmd)
	analogyCmd.Flags().StringVarP(&analogyModelFile, "model", "m", "vectors.bin", "Path to the word vector model file")
	analogyCmd.Flags().StringSliceVar(&analogyWords, "words", []string{}, "A comma-separated list of three words for the analogy task (e.g., king,man,woman)")
	analogyCmd.Flags().IntVarP(&analogyTopN, "topn", "n", 10, "Number of closest words to find")
	analogyCmd.MarkFlagRequired("words")
}

func runAnalogy(cmd *cobra.Command, args []string) {
	if len(analogyWords) != 3 {
		log.Fatal("Exactly three words are required for the analogy task.")
	}

	m, err := word2vec.Load(analogyModelFile)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	log.Printf("Loaded model.")

	// Get vectors for the three words
	vecs := make([][]float32, 3)
	for i, word := range analogyWords {
		vecs[i] = make([]float32, m.Size())
		if err := m.VectorOf(word, vecs[i]); err != nil {
			log.Fatalf("Failed to get vector for word '%s': %v", word, err)
		}
	}

	// Perform the vector arithmetic: word1 - word2 + word3
	resultVector := make([]float32, m.Size())
	for i := 0; i < m.Size(); i++ {
		resultVector[i] = vecs[0][i] - vecs[1][i] + vecs[2][i]
	}

	// Create a map of words to exclude from the search results.
	// We exclude the original words used in the analogy.
	exclude := make(map[string]bool)
	for _, word := range analogyWords {
		// The public API of the model handles stemming, so we just pass the raw word.
		exclude[m.ProcessWord(word)] = true
	}

	// Find the nearest words to the resulting vector
	results := make([]word2vec.Nearest, analogyTopN)
	if err := m.NearestToVector(resultVector, results, exclude); err != nil {
		log.Fatalf("Analogy lookup failed: %v", err)
	}

	fmt.Printf("\nAnalogy: %s - %s + %s = ?\n", analogyWords[0], analogyWords[1], analogyWords[2])
	for _, res := range results {
		fmt.Printf("%20s\t%f\n", res.Word, res.Distance)
	}
}
