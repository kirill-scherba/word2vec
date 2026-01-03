// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import (
	"bufio"
	"io"
	"os"
)

// WordReader is responsible for reading words from an input stream (a text file).
type WordReader struct {
	scanner    *bufio.Scanner
	file       *os.File
	wordCount  int64
	totalBytes int64
}

// NewWordReader creates a new WordReader for the given file path.
func NewWordReader(filePath string) (*WordReader, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	stat, err := file.Stat()
	if err != nil {
		return nil, err
	}

	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanWords)

	return &WordReader{
		scanner:    scanner,
		file:       file,
		totalBytes: stat.Size(),
	}, nil
}

// ReadWord reads the next word from the stream.
// It returns the word and an error. io.EOF is returned when the stream ends.
func (r *WordReader) ReadWord() (string, error) {
	if r.scanner.Scan() {
		r.wordCount++
		return CleanWord(r.scanner.Text()), nil
	}
	return "", io.EOF
}

// Close closes the underlying file reader.
func (r *WordReader) Close() error {
	return r.file.Close()
}

// Seek moves the file pointer to a given offset and resets the scanner.
// It then reads until the next whitespace to align to a word boundary.
func (r *WordReader) Seek(offset int64, whence int) (int64, error) {
	newOffset, err := r.file.Seek(offset, whence)
	if err != nil {
		return 0, err
	}

	// Reset the scanner to read from the new position.
	// We need to read until the first word boundary to avoid starting mid-word.
	if newOffset > 0 {
		r.scanner = bufio.NewScanner(r.file)
		r.scanner.Split(bufio.ScanWords)
		r.scanner.Scan() // Discard the potentially partial first word.
	}

	return newOffset, nil
}
