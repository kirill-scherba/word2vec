// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sort"
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
	StemMap() map[string]string
}

// SaveModel writes the model vectors to a writer in the word2vec binary format.
// It adds a custom header byte to indicate if stemming was used.
func SaveModel(model ModelProvider, vocab VocabProvider, lemmatize bool, w io.Writer) error {
	if f, ok := w.(io.Closer); ok {
		defer f.Close()
	}

	writer := bufio.NewWriter(w)
	defer writer.Flush()

	vectors := model.ExportSyn0()
	if len(vectors) == 0 {
		return fmt.Errorf("model has no vectors to save")
	}

	// Custom Header: Write a single byte to indicate if stemming was used.
	var lemmatizeByte byte = 0
	if lemmatize {
		lemmatizeByte = 1
	}
	if err := binary.Write(writer, binary.LittleEndian, lemmatizeByte); err != nil {
		return fmt.Errorf("failed to write lemmatize flag: %w", err)
	}

	vocabSize := vocab.Len()
	vectorSize := len(vectors[0])

	// Write header
	fmt.Fprintf(writer, "%d %d\n", vocabSize, vectorSize)

	// Write stem map only if lemmatization was used
	if lemmatize {
		stemMap := vocab.StemMap()
		fmt.Fprintf(writer, "%d\n", len(stemMap))
		if len(stemMap) > 0 {
			// Sort for deterministic output
			stems := make([]string, 0, len(stemMap))
			for stem := range stemMap {
				stems = append(stems, stem)
			}
			sort.Strings(stems)
			for _, stem := range stems {
				fmt.Fprintf(writer, "%s %s\n", stem, stemMap[stem])
			}
		}
	}

	// Write words and vectors
	for i, word := range vocab.Words() {
		fmt.Fprintf(writer, "%s ", word.Word)
		if err := binary.Write(writer, binary.LittleEndian, vectors[i]); err != nil {
			return fmt.Errorf("failed to write vector for word %s: %w", word.Word, err)
		}
		writer.WriteByte('\n')
	}
	return nil
}

// LoadModel reads a model from a reader.
func LoadModel(r io.Reader) (words []string, vectors [][]float32, lemmatized bool, stemMap map[string]string, err error) {
	reader := bufio.NewReader(r)

	// Custom Header: Read the lemmatize flag.
	lemmatizeByte, err := reader.ReadByte()
	if err != nil {
		return nil, nil, false, nil, fmt.Errorf("failed to read model file: %w", err)
	}

	if lemmatizeByte != 0 && lemmatizeByte != 1 {
		// This is likely an old model file without our custom header.
		// We need to "unread" the byte and assume lemmatization is false.
		if err := reader.UnreadByte(); err != nil {
			return nil, nil, false, nil, fmt.Errorf("failed to unread byte for legacy model: %w", err)
		}
		lemmatized = false
	} else {
		lemmatized = (lemmatizeByte == 1)
	}

	var vocabSize, vectorSize int
	_, err = fmt.Fscanf(reader, "%d %d\n", &vocabSize, &vectorSize)
	if err != nil {
		return nil, nil, false, nil, fmt.Errorf("failed to read model header: %w", err)
	}
	if vocabSize == 0 || vectorSize == 0 {
		return nil, nil, false, nil, errors.New("invalid model file: vocab or vector size is zero")
	}

	// Read stem map
	stemMap = make(map[string]string)
	if lemmatized {
		var stemMapSize int
		_, err = fmt.Fscanf(reader, "%d\n", &stemMapSize)
		if err != nil {
			return nil, nil, false, nil, fmt.Errorf("failed to read stem map size: %w", err)
		}
		for i := 0; i < stemMapSize; i++ {
			var stem, original string
			line, err := reader.ReadString('\n')
			if err != nil {
				return nil, nil, false, nil, fmt.Errorf("failed to read stem map entry: %w", err)
			}
			parts := strings.Fields(line)
			if len(parts) == 2 {
				stem, original = parts[0], parts[1]
			}
			stemMap[stem] = original
		}
	}

	words = make([]string, vocabSize)
	vectors = make([][]float32, vocabSize)

	for i := 0; i < vocabSize; i++ {
		word, err := reader.ReadString(' ')
		if err != nil {
			return nil, nil, false, nil, fmt.Errorf("failed to read word on line %d: %w", i, err)
		}
		words[i] = strings.TrimSpace(word)

		vec := make([]float32, vectorSize)
		if err := binary.Read(reader, binary.LittleEndian, &vec); err != nil {
			return nil, nil, false, nil, fmt.Errorf("failed to read vector on line %d: %w", i, err)
		}
		vectors[i] = vec

		reader.ReadByte() // Read and discard the trailing newline character.
	}

	return words, vectors, lemmatized, stemMap, nil
}
