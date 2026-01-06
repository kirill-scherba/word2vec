// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package word2vec

import (
	"fmt"
	"log"
	"os"

	"github.com/kirill-scherba/word2vec/internal/libw2v"
)

// TrainConfig holds all the parameters for training a new model.
type TrainConfig = libw2v.TrainSettings

// Train starts the model training process based on the provided configuration.
func Train(config TrainConfig, trainFile, outFile string) (err error) {

	// Check if output file is specified
	if outFile == "" {
		err = fmt.Errorf("Output file is not specified")
		return
	}

	// Initialize trainer
	trainer, err := libw2v.NewTrainer(config, trainFile)
	if err != nil {
		err = fmt.Errorf("Failed to initialize trainer: %v", err)
		return
	}

	// Start training
	model := trainer.Train()

	// Save model
	f, err := os.Create(outFile)
	if err != nil {
		err = fmt.Errorf("Failed to create output file: %v", err)
		return
	}
	err = libw2v.SaveModel(model, trainer.Vocab(), config.Lemmatize, f)
	if err != nil {
		err = fmt.Errorf("Failed to save model: %v", err)
		return
	}
	log.Printf("Model saved to %s", outFile)

	return
}
