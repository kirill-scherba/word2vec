// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package word2vec

import "github.com/kirill-scherba/word2vec/internal/libw2v"

// Dot calculates the dot product of two vectors.
func Dot(v1, v2 []float32) float32 {
	return libw2v.Dot(v1, v2)
}

// Normalize normalizes a vector to unit length.
func Normalize(v []float32) {
	libw2v.Normalize(v)
}

// IsStopWord checks if a word is a stop word (currently for Russian).
func IsStopWord(word string) bool {
	return libw2v.IsStopWord(word)
}

// CleanWord converts a word to lower case and removes punctuation and symbols.
func CleanWord(word string) string {
	return libw2v.CleanWord(word)
}
