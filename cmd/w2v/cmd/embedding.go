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
	embModelFile string
	embText      string
)

var embeddingCmd = &cobra.Command{
	Use:   "embedding",
	Short: "Calculate embedding for a document (text)",
	Run:   runEmbedding,
}

func init() {
	rootCmd.AddCommand(embeddingCmd)
	embeddingCmd.Flags().StringVarP(&embModelFile, "model", "m", "vectors.bin", "Path to the word vector model file")
	embeddingCmd.Flags().StringVarP(&embText, "text", "t", "", "Text to calculate embedding for (required)")
	embeddingCmd.MarkFlagRequired("text")
}

func runEmbedding(cmd *cobra.Command, args []string) {
	m, err := word2vec.Load(embModelFile)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	log.Printf("Loaded model.")

	vector := make([]float32, m.Size())
	if err := m.Embedding(embText, vector); err != nil {
		log.Fatalf("Failed to calculate embedding: %v", err)
	}

	fmt.Println("Embedding vector:")
	fmt.Println(vector)
}
