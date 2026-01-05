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

var similarityCmd = &cobra.Command{
	Use:   "similarity",
	Short: "Calculate the cosine similarity between two texts",
	Long: `Loads a pre-trained model and calculates the cosine similarity
between two pieces of text provided via flags. A score of 1 means the texts are
very similar, and a score of -1 means they are very dissimilar.`,
	Run: runSimilarity,
}

func init() {
	rootCmd.AddCommand(similarityCmd)
	similarityCmd.Flags().String("model", "vectors.bin", "Path to the pre-trained model file")
	similarityCmd.Flags().String("text1", "", "First text to compare")
	similarityCmd.Flags().String("text2", "", "Second text to compare")
	similarityCmd.MarkFlagRequired("model")
	similarityCmd.MarkFlagRequired("text1")
	similarityCmd.MarkFlagRequired("text2")
}

func runSimilarity(cmd *cobra.Command, args []string) {
	modelFile, _ := cmd.Flags().GetString("model")
	text1, _ := cmd.Flags().GetString("text1")
	text2, _ := cmd.Flags().GetString("text2")

	model, err := word2vec.Load(modelFile)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	similarity, err := model.Similarity(text1, text2)
	if err != nil {
		log.Fatalf("Failed to calculate similarity: %v", err)
	}

	fmt.Printf("Similarity: %f\n", similarity)
}
