// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"log"
	"os"
	"runtime/pprof"

	"github.com/kirill-scherba/word2vec"
	"github.com/kirill-scherba/word2vec/internal/libw2v"
	"github.com/spf13/cobra"
)

var (
	trainFile       string
	outFile         string
	cpuprofile      string
	vectorSize      int
	windowSize      int
	minCount        int
	numThreads      int
	numEpochs       int
	learningRate    float64
	subsample       float64
	useCbow         bool
	negativeSamples int
	useHS           bool
	verbose         bool
	lemmatize       bool
)

var trainCmd = &cobra.Command{
	Use:   "train",
	Short: "Train a new model",
	Run:   runTrain,
}

func init() {
	rootCmd.AddCommand(trainCmd)
	trainCmd.Flags().StringVarP(&trainFile, "train", "", "", "Path to the training text file (required)")
	trainCmd.Flags().StringVarP(&outFile, "output", "o", "vectors.bin", "Path to save the resulting word vectors")
	trainCmd.Flags().StringVarP(&cpuprofile, "cpuprofile", "", "", "Write cpu profile to this file")
	trainCmd.Flags().IntVarP(&vectorSize, "size", "", 100, "Set size of word vectors")
	trainCmd.Flags().IntVarP(&windowSize, "window", "", 5, "Set max skip length between words")
	trainCmd.Flags().IntVarP(&minCount, "min-count", "", 5, "Discard words that appear less than <int> times")
	trainCmd.Flags().IntVarP(&numThreads, "threads", "", 12, "Use <int> threads")
	trainCmd.Flags().IntVarP(&numEpochs, "iter", "", 5, "Run more training iterations")
	trainCmd.Flags().Float64VarP(&learningRate, "alpha", "", 0.025, "Set the starting learning rate")
	trainCmd.Flags().Float64VarP(&subsample, "sample", "", 1e-3, "Set threshold for occurrence of words")
	trainCmd.Flags().BoolVarP(&useCbow, "cbow", "", false, "Use the continuous bag of words model; default is skip-gram")
	trainCmd.Flags().IntVarP(&negativeSamples, "negative", "", 5, "Number of negative examples (0 = not used)")
	trainCmd.Flags().BoolVarP(&useHS, "hs", "", false, "Use Hierarchical Softmax")
	trainCmd.Flags().BoolVarP(&verbose, "verbose", "", true, "Enable verbose output")
	trainCmd.Flags().BoolVarP(&lemmatize, "lemmatize", "l", false, "Use lemmatization for Russian text")
	trainCmd.MarkFlagRequired("train")
}

func runTrain(cmd *cobra.Command, args []string) {
	if cpuprofile != "" {
		f, err := os.Create(cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	// Train config
	settings := word2vec.TrainConfig{
		VectorSize:           vectorSize,
		WindowSize:           windowSize,
		MinCount:             minCount,
		NumThreads:           numThreads,
		NumEpochs:            numEpochs,
		LearningRate:         float32(learningRate),
		SubsamplingThreshold: float32(subsample),
		NegativeSamples:      negativeSamples,
		Verbose:              verbose,
		Lemmatize:            lemmatize,
	}
	if useCbow {
		settings.Architecture = libw2v.CBOW
	} else {
		settings.Architecture = libw2v.SkipGram
	}
	if useHS {
		settings.LossFunction = libw2v.HierarchicalSoftmax
		settings.NegativeSamples = 0
	} else {
		settings.LossFunction = libw2v.NegativeSampling
	}

	// Train
	err := word2vec.Train(settings, trainFile, outFile)
	if err != nil {
		log.Fatalf("Failed to train model: %v", err)
	}
}
