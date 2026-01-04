// Copyright 2025 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package libw2v contains a Russian stemmer based on the Porter algorithm.

package libw2v

import (
	"regexp"
	"strings"
)

var (
	rvre = regexp.MustCompile(`^(.*?[аеиоуыэюя])(.*)$`)

	// stemmerExceptions is a map of words that should not be stemmed to avoid
	// incorrect root detection (e.g., "манго" -> "манг" <- "мангал").
	stemmerExceptions = map[string]struct{}{
		"манго": {},
	}

	perfectiveGround1 = regexp.MustCompile(`(ив|ивши|ившись|ыв|ывши|ывшись)$`)
	perfectiveGround2 = regexp.MustCompile(`([ая])(в|вши|вшись)$`)

	reflexive = regexp.MustCompile(`(с[яь])$`)

	adjective = regexp.MustCompile(`(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$`)

	participle1 = regexp.MustCompile(`(ивш|ывш|ующ)$`)
	participle2 = regexp.MustCompile(`([ая])(ем|нн|вш|ющ|щ)$`)

	verb1 = regexp.MustCompile(`(ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)$`)
	verb2 = regexp.MustCompile(`([ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)$`)

	noun = regexp.MustCompile(`(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)$`)

	superlative = regexp.MustCompile(`(ейше|ейш)$`)

	derivational = regexp.MustCompile(`ость?$`)

	i  = regexp.MustCompile(`и$`)
	p  = regexp.MustCompile(`ь$`)
	nn = regexp.MustCompile(`нн$`)
)

// Stem applies the Porter Stemmer algorithm to a Russian word.
func Stem(word string) string {

	// return word

	// Do not stem short words
	if len([]rune(word)) <= 4 {
		return word
	}

	word = strings.ToLower(word)
	word = strings.Replace(word, "ё", "е", -1)

	// Check for exceptions before stemming
	if _, ok := stemmerExceptions[word]; ok {
		return word
	}

	m := rvre.FindStringSubmatch(word)
	if len(m) < 3 {
		return word
	}

	pre := m[1]
	rv := m[2]

	// Step 1
	temp := perfectiveGround1.ReplaceAllString(rv, "")
	if len(temp) == len(rv) {
		temp = perfectiveGround2.ReplaceAllString(rv, "$1")
	}

	if len(temp) != len(rv) {
		rv = temp
	} else {
		rv = reflexive.ReplaceAllString(rv, "")
		temp = adjective.ReplaceAllString(rv, "")
		if len(temp) != len(rv) {
			rv = temp
			temp = participle1.ReplaceAllString(rv, "")
			if len(temp) == len(rv) {
				temp = participle2.ReplaceAllString(rv, "$1")
			}
			rv = temp
		} else {
			temp = verb1.ReplaceAllString(rv, "")
			if len(temp) == len(rv) {
				temp = verb2.ReplaceAllString(rv, "$1")
			}
			if len(temp) == len(rv) {
				rv = noun.ReplaceAllString(rv, "")
			} else {
				rv = temp
			}
		}
	}

	// Step 2
	rv = i.ReplaceAllString(rv, "")

	// Step 3
	if derivational.MatchString(rv) {
		rv = derivational.ReplaceAllString(rv, "")
	}

	// Step 4
	// Don't remove the soft sign 'ь' at the end, as it's often part of the stem.
	rv = superlative.ReplaceAllString(rv, "")
	rv = nn.ReplaceAllString(rv, "н")
	rv = p.ReplaceAllString(rv, "") // Move soft sign removal to the very end.

	return pre + rv
}
