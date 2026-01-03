// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import (
	"io"
	"sort"
)

// VocabWord represents a word in the vocabulary.
type VocabWord struct {
	Word  string
	Count int64
}

// Vocabulary holds the words from the training corpus, their counts,
// and a mapping for quick lookups.
type Vocabulary struct {
	words      []VocabWord
	wordMap    map[string]int
	trainWords int64
}

// NewVocabulary builds a vocabulary from a corpus using a WordReader.
// It counts word frequencies, discards rare words based on settings.MinCount,
// and sorts the vocabulary by frequency.
func NewVocabulary(reader *WordReader, settings TrainSettings) (*Vocabulary, error) {
	freqMap := make(map[string]int64)
	trainWords := int64(0)

	// 1. Read the corpus and count word frequencies.
	for {
		word, err := reader.ReadWord()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		freqMap[word]++
	}

	// 2. Filter words that are less frequent than MinCount.
	vocabWords := make([]VocabWord, 0, len(freqMap))
	for word, count := range freqMap {
		if count >= int64(settings.MinCount) {
			vocabWords = append(vocabWords, VocabWord{Word: word, Count: count})
			trainWords += count
		}
	}

	// 3. Sort the vocabulary by frequency in descending order.
	sort.Slice(vocabWords, func(i, j int) bool {
		return vocabWords[i].Count > vocabWords[j].Count
	})

	// 4. Create a map for fast word-to-index lookups.
	wordMap := make(map[string]int, len(vocabWords))
	for i, v := range vocabWords {
		wordMap[v.Word] = i
	}

	return &Vocabulary{
		words:      vocabWords,
		wordMap:    wordMap,
		trainWords: trainWords,
	}, nil
}

// Words returns the sorted list of vocabulary words.
func (v *Vocabulary) Words() []VocabWord {
	return v.words
}

// Len returns the number of unique words in the vocabulary.
func (v *Vocabulary) Len() int {
	return len(v.words)
}
