// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import (
	"strings"
	"unicode"
)

// CleanWord converts a word to lower case and removes punctuation and symbols.
func CleanWord(word string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) || unicode.IsSymbol(r) || unicode.IsNumber(r) {
			return -1
		}
		return unicode.ToLower(r)
	}, word)
}
