// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import "math"

// Dot calculates the dot product of two vectors.
func Dot(v1, v2 []float32) float32 {
	var res float32
	for i := range v1 {
		res += v1[i] * v2[i]
	}
	return res
}

// Normalize normalizes a vector to unit length.
func Normalize(v []float32) {
	var norm float32
	for _, val := range v {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}
