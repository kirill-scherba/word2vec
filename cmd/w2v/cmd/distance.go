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
	distModelFile string
	distWord      string
	distTopN      int
)

var distanceCmd = &cobra.Command{
	Use:   "distance",
	Short: "Find closest words for a given word",
	Run:   runDistance,
}

func init() {
	rootCmd.AddCommand(distanceCmd)
	distanceCmd.Flags().StringVarP(&distModelFile, "model", "m", "vectors.bin", "Path to the word vector model file")
	distanceCmd.Flags().StringVarP(&distWord, "word", "w", "", "Word to find distance for (required)")
	distanceCmd.Flags().IntVarP(&distTopN, "topn", "n", 10, "Number of closest words to find")
	distanceCmd.MarkFlagRequired("word")
}

func runDistance(cmd *cobra.Command, args []string) {
	m, err := word2vec.Load(distModelFile)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	log.Printf("Loaded model.")

	results := make([]word2vec.Nearest, distTopN)
	if err := m.Lookup(distWord, results); err != nil {
		log.Fatalf("Lookup failed: %v", err)
	}

	fmt.Printf("\nWords closest to '%s':\n", distWord)
	for _, res := range results {
		if res.Word == "" {
			break
		}
		fmt.Printf("%20s\t%f\n", res.Word, res.Distance)
	}
}
