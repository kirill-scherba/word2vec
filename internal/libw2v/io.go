// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"strings"
)

// ModelProvider defines an interface for objects that can provide word vectors.
type ModelProvider interface {
	ExportSyn0() [][]float32
}

// VocabProvider defines an interface for objects that can provide vocabulary words.
type VocabProvider interface {
	Words() []VocabWord
	Len() int
}

// SaveModel writes the model vectors to a writer in the word2vec binary format.
func SaveModel(model ModelProvider, vocab VocabProvider, w io.Writer) error {
	if f, ok := w.(io.Closer); ok {
		defer f.Close()
	}

	writer := bufio.NewWriter(w)

	vectors := model.ExportSyn0()
	if len(vectors) == 0 {
		return fmt.Errorf("model has no vectors to save")
	}
	vocabSize := vocab.Len()
	vectorSize := len(vectors[0])

	// Write header
	fmt.Fprintf(writer, "%d %d\n", vocabSize, vectorSize)

	// Write words and vectors
	for i, word := range vocab.Words() {
		fmt.Fprintf(writer, "%s ", word.Word)
		if err := binary.Write(writer, binary.LittleEndian, vectors[i]); err != nil {
			return fmt.Errorf("failed to write vector for word %s: %w", word.Word, err)
		}
		writer.WriteByte('\n')
	}
	return writer.Flush()
}

// LoadModel reads a model from a reader.
func LoadModel(r io.Reader) (words []string, vectors [][]float32, err error) {
	reader := bufio.NewReader(r)

	var vocabSize, vectorSize int
	_, err = fmt.Fscanf(reader, "%d %d\n", &vocabSize, &vectorSize)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read model header: %w", err)
	}

	words = make([]string, vocabSize)
	vectors = make([][]float32, vocabSize)

	for i := 0; i < vocabSize; i++ {
		word, err := reader.ReadString(' ')
		if err != nil {
			return nil, nil, fmt.Errorf("failed to read word on line %d: %w", i, err)
		}
		words[i] = strings.TrimSpace(word)

		vec := make([]float32, vectorSize)
		if err := binary.Read(reader, binary.LittleEndian, &vec); err != nil {
			return nil, nil, fmt.Errorf("failed to read vector on line %d: %w", i, err)
		}
		vectors[i] = vec

		reader.ReadByte() // Read and discard the trailing newline character.
	}

	return words, vectors, nil
}
