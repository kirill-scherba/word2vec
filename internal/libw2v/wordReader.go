// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import (
	"bufio"
	"fmt"
	"io"
	"os"
)

// WordReader is responsible for reading words from an input stream (a text file).
type WordReader struct {
	scanner *bufio.Scanner
	file    *os.File // Keep the original file to get position
	reader  io.ReadCloser
}

// NewWordReaderFromPath creates a new WordReader for the given file path.
func NewWordReaderFromPath(filePath string) (*WordReader, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, err
	}
	_ = stat // We don't need totalBytes here anymore

	return NewWordReader(file, file), nil
}

// NewWordReader creates a new WordReader from an io.ReadCloser.
func NewWordReader(r io.ReadCloser, f *os.File) *WordReader {
	scanner := bufio.NewScanner(r)
	scanner.Split(bufio.ScanWords)
	return &WordReader{scanner: scanner, reader: r, file: f}
}

// ReadWord reads the next word from the stream. It's the caller's responsibility to track the word count.
// It returns the word and an error. io.EOF is returned when the stream ends.
func (r *WordReader) ReadWord() (word string, bytesRead int64, err error) {
	if r.scanner.Scan() {
		word = r.scanner.Text()
		bytesRead = int64(len(word)) // Return word length as an approximation
		return CleanWord(word), bytesRead, nil
	}
	if err := r.scanner.Err(); err != nil {
		return "", 0, err
	}
	return "", 0, io.EOF // Correctly signal end of file
}

// Close closes the underlying file reader.
func (r *WordReader) Close() error {
	if r.reader != nil {
		return r.reader.Close()
	}
	return nil
}

// Pos returns the current reading position in the underlying file.
func (r *WordReader) Pos() (int64, error) {
	if r.file == nil {
		return 0, fmt.Errorf("underlying file is not available to get position")
	}
	return r.file.Seek(0, io.SeekCurrent)
}

// GetFileSize returns the size of the file.
func GetFileSize(filePath string) (int64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return 0, err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return 0, err
	}
	return stat.Size(), nil
}
