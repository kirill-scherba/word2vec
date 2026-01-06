// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import (
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

const expTableSize = 1000
const maxExp = 6

// Trainer orchestrates the model training process.
type Trainer struct {
	settings  TrainSettings
	vocab     *Vocabulary
	model     *Model
	sampler   *Sampler
	huffman   *HuffmanTree
	trainFile string

	// expTable is a precomputed table for the sigmoid function.
	expTable []float32

	// Atomic counters for tracking progress across goroutines.
	wordCountActual *atomic.Int64
	alpha           *atomic.Uint32 // Stored as float32 bits
	bytesRead       *atomic.Int64  // Track bytes read for progress
	done            chan struct{}  // Channel to signal completion
}

// NewTrainer sets up a new training session.
func NewTrainer(settings TrainSettings, trainFile string) (*Trainer, error) {

	// Step 0: Create word reader
	reader, err := NewWordReaderFromPath(trainFile)
	if err != nil {
		return nil, fmt.Errorf("failed to create word reader: %w", err)
	}
	defer reader.Close()

	// Step 1: Build Vocabulary
	if settings.Verbose {
		log.Printf("Build Vocabulary...\n")
	}
	vocab, err := NewVocabulary(reader, settings)
	if err != nil {
		return nil, fmt.Errorf("failed to build vocabulary: %w", err)
	}
	if settings.Verbose {
		log.Printf("Vocabulary size: %d\n", vocab.Len())
		log.Printf("Words in train file: %d\n", vocab.trainWords)
	}

	// Step 2: Initialize components based on settings
	model := NewModel(vocab, settings)
	var huffman *HuffmanTree
	var sampler *Sampler

	if settings.LossFunction == HierarchicalSoftmax {
		huffman = NewHuffmanTree(vocab)
	}
	if settings.NegativeSamples > 0 {
		sampler = NewSampler(vocab)
	}

	// Step 3: Precompute exponent table
	expTable := make([]float32, expTableSize)
	for i := range expTableSize {
		// Precompute the exp() table
		exp := math.Exp(float64(i)/float64(expTableSize)*2*maxExp - maxExp)
		// Precompute f(x) = x / (x + 1)
		expTable[i] = float32(exp / (exp + 1))
	}

	return &Trainer{
		settings:        settings,
		vocab:           vocab,
		model:           model,
		sampler:         sampler,
		huffman:         huffman,
		trainFile:       trainFile,
		expTable:        expTable,
		wordCountActual: &atomic.Int64{},
		alpha:           &atomic.Uint32{},
		bytesRead:       &atomic.Int64{},
		done:            make(chan struct{}),
	}, nil
}

// Train starts the training process.
func (t *Trainer) Train() *Model {
	t.alpha.Store(math.Float32bits(t.settings.LearningRate))

	if t.settings.Verbose {
		log.Println("Starting training...")
	}

	var workerWg sync.WaitGroup
	// Workers are now started inside produceSentences

	// Progress reporting thread
	if t.settings.Verbose {
		var reporterWg sync.WaitGroup
		reporterWg.Add(1)
		go t.reportProgress(&reporterWg)

		// Wait for training workers to finish, then signal the reporter to stop.
		t.produceSentences(&workerWg) // This will block until all epochs are done
		close(t.done)
		reporterWg.Wait()
	} else {
		// If not verbose, just wait for workers to finish.
		t.produceSentences(&workerWg)
	}

	if t.settings.Verbose {
		log.Println("\nTraining finished.")
	}

	return t.model
}

// produceSentences is the single producer goroutine. It reads the training file
// and sends sentences (slices of word indices) to the worker goroutines via a channel.
func (t *Trainer) produceSentences(wg *sync.WaitGroup) {
	sentenceChan := make(chan []int, t.settings.NumThreads*2)

	// Start consumer goroutines (the workers)
	for i := 0; i < t.settings.NumThreads; i++ {
		wg.Add(1)
		go t.trainWorker(i, wg, sentenceChan)
	}

	// Start producing sentences for each epoch
	for epoch := 0; epoch < t.settings.NumEpochs; epoch++ {
		reader, err := NewWordReaderFromPath(t.trainFile)
		if err != nil {
			log.Printf("Epoch %d: failed to open training file: %v", epoch, err)
			break // Stop if we can't read the file
		}

		sentence := make([]int, 0, 1000)
		var lastPos int64
		for {
			wordStr, _, err := reader.ReadWord()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Printf("Epoch %d: error reading word: %v", epoch, err)
				break
			}
			pos, err := reader.Pos()
			if err != nil {
				log.Printf("Epoch %d: error getting position: %v", epoch, err)
				break
			}
			t.bytesRead.Add(pos - lastPos)
			lastPos = pos

			if wordIndex, exists := t.vocab.wordMap[wordStr]; exists {
				sentence = append(sentence, wordIndex)
			}

			if len(sentence) >= 1000 {
				sentenceChan <- sentence
				sentence = make([]int, 0, 1000)
			}
		}
		reader.Close()
	}
	close(sentenceChan) // Close channel to signal workers there's no more work
	wg.Wait()           // Wait for all workers to finish processing
}

// Vocab returns the vocabulary used by the trainer.
func (t *Trainer) Vocab() *Vocabulary {
	return t.vocab
}

// reportProgress periodically logs the training progress.
func (t *Trainer) reportProgress(wg *sync.WaitGroup) {
	defer wg.Done()
	fileSize, err := GetFileSize(t.trainFile)
	if err != nil {
		log.Printf("Error getting file size for progress reporting: %v", err)
		return
	}
	totalBytesToProcess := fileSize * int64(t.settings.NumEpochs)
	startTime := time.Now()

	ticker := time.NewTicker(1000 * time.Millisecond)
	defer ticker.Stop()

	printLine := func(final bool) {
		bytes := t.bytesRead.Load()
		progress := float64(bytes) / float64(totalBytesToProcess) * 100
		if progress > 100 {
			progress = 100
		}

		currentAlpha := math.Float32frombits(t.alpha.Load())
		elapsed := time.Since(startTime).Seconds()
		wps := float64(t.wordCountActual.Load()) / elapsed

		format := "\rAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk"
		if final {
			format += "\n"
		}
		log.Printf(format, currentAlpha, progress, wps/1000)
	}

	for {
		select {
		case <-t.done:
			// Training is complete, do one final print and exit.
			printLine(true)
			return
		case <-ticker.C:
			// Update learning rate (alpha) based on the actual progress.
			bytes := t.bytesRead.Load()
			currentAlpha := t.settings.LearningRate * (1.0 - float32(bytes)/float32(totalBytesToProcess+1))
			if currentAlpha < t.settings.LearningRate*0.0001 {
				currentAlpha = t.settings.LearningRate * 0.0001
			}
			t.alpha.Store(math.Float32bits(currentAlpha))
			printLine(false)
		}
	}
}

// trainWorker is the main function for each training thread (goroutine).
func (t *Trainer) trainWorker(id int, wg *sync.WaitGroup, sentenceChan <-chan []int) {
	defer wg.Done()

	// Each goroutine gets its own random number generator to avoid lock contention.
	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(id)))

	// Allocate a buffer for gradient updates once per worker.
	neu1e := make([]float32, t.settings.VectorSize)
	localWordCount := int64(0)

	// Consume sentences from the channel until it's closed
	for sentence := range sentenceChan {
		t.processSentence(sentence, &localWordCount, rng, neu1e)
	}
}

// processSentence runs the training algorithm on a single sentence.
func (t *Trainer) processSentence(sentence []int, localWordCount *int64, rng *rand.Rand, neu1e []float32) {
	alpha := math.Float32frombits(t.alpha.Load())

	for pos, wordIndex := range sentence {
		// Update learning rate
		// The learning rate is now updated centrally by the progress reporter
		// to prevent race conditions and ensure smooth decay.
		alpha = math.Float32frombits(t.alpha.Load())
		// Generate random window size
		b := rng.Intn(t.settings.WindowSize)

		if t.settings.Architecture == SkipGram {
			// Train Skip-Gram
			for a := b; a < t.settings.WindowSize*2+1-b; a++ {
				if a == t.settings.WindowSize {
					continue
				}
				c := pos - t.settings.WindowSize + a
				if c < 0 || c >= len(sentence) {
					continue
				}

				contextWordIndex := sentence[c]
				if contextWordIndex == wordIndex {
					continue
				}

				if t.settings.LossFunction == NegativeSampling {
					t.trainSkipGramNegativeSampling(wordIndex, contextWordIndex, alpha, rng, neu1e)
				} else {
					t.trainSkipGramHierarchicalSoftmax(wordIndex, contextWordIndex, alpha, neu1e)
				}
			}
		} else { // CBOW
			// Train CBOW
			context, contextLen := t.getContext(sentence, pos, b)
			if contextLen == 0 {
				continue
			}

			// Dispatch to the correct training algorithm
			if t.settings.LossFunction == NegativeSampling {
				t.trainCBOWNegativeSampling(wordIndex, context, alpha, rng, neu1e)
			} else {
				t.trainCBOWHierarchicalSoftmax(wordIndex, context, alpha, neu1e)
			}
		}
		*localWordCount++
		t.wordCountActual.Add(1) // Increment the global counter for each processed word
	}
}

func (t *Trainer) getContext(sentence []int, pos, b int) ([]int, int) {
	context := make([]int, 0, t.settings.WindowSize*2)
	contextLen := 0
	for a := b; a < t.settings.WindowSize*2+1-b; a++ {
		if a == t.settings.WindowSize {
			continue
		}
		c := pos - t.settings.WindowSize + a
		if c < 0 || c >= len(sentence) {
			continue
		}
		context = append(context, sentence[c])
		contextLen++
	}
	return context, contextLen
}

func (t *Trainer) sigmoid(f float32) float32 {
	if f > float32(maxExp) {
		return 1.0
	}
	if f < -float32(maxExp) {
		return 0.0
	}
	return t.expTable[int((f+float32(maxExp))*(float32(expTableSize)/float32(maxExp)/2.0))]
}

// trainSkipGramNegativeSampling implements the Skip-gram model with negative sampling.
func (t *Trainer) trainSkipGramNegativeSampling(wordIndex, contextWordIndex int, alpha float32, rng *rand.Rand, neu1e []float32) {
	vectorSize := t.settings.VectorSize
	syn0 := t.model.syn0[wordIndex]
	syn1neg := t.model.syn1neg

	// Clear the gradient buffer
	for i := range vectorSize {
		neu1e[i] = 0
	}

	var target, label int
	for d := 0; d <= t.settings.NegativeSamples; d++ {
		if d == 0 {
			// Positive sample
			target = contextWordIndex
			label = 1
		} else {
			// Negative sample
			target = t.sampler.unigramTable[rng.Intn(unigramTableSize)]
			if target == contextWordIndex {
				continue // Skip if we accidentally sample the positive word
			}
			label = 0
		}

		targetVector := syn1neg[target]

		// Calculate dot product between input word vector and target vector
		dot := float32(0.0)
		for i := range vectorSize {
			dot += syn0[i] * targetVector[i]
		}

		// Calculate prediction and error
		prediction := t.sigmoid(dot)
		gradient := alpha * (float32(label) - prediction)

		// Update the gradient buffer (for updating the input word later)
		for i := range vectorSize {
			neu1e[i] += gradient * targetVector[i]
		}

		// Update the target word's vector (from the negative sampling matrix)
		for i := range vectorSize {
			targetVector[i] += gradient * syn0[i]
		}
	}

	// Update the input word's vector using the accumulated gradients
	t.model.mu.Lock()
	for i := range vectorSize {
		syn0[i] += neu1e[i]
	}
	t.model.mu.Unlock()
}

// trainCBOWNegativeSampling implements the CBOW model with negative sampling.
func (t *Trainer) trainCBOWNegativeSampling(wordIndex int, context []int, alpha float32, rng *rand.Rand, neu1e []float32) {
	t.model.mu.Lock()
	defer t.model.mu.Unlock()

	vectorSize := t.settings.VectorSize
	syn0 := t.model.syn0
	syn1neg := t.model.syn1neg
	neu1 := make([]float32, vectorSize) // Hidden layer activations

	// Sum the vectors of the context words
	for _, contextWordIndex := range context {
		contextVector := syn0[contextWordIndex]
		for i := range vectorSize {
			neu1[i] += contextVector[i]
		}
	}

	// Average the context vectors
	contextLen := float32(len(context))
	for i := range vectorSize {
		neu1[i] /= contextLen
	}

	// Clear the gradient buffer
	for i := range vectorSize {
		neu1e[i] = 0
	}

	var target, label int
	for d := 0; d <= t.settings.NegativeSamples; d++ {
		if d == 0 {
			// Positive sample
			target = wordIndex
			label = 1
		} else {
			// Negative sample
			target = t.sampler.unigramTable[rng.Intn(unigramTableSize)]
			if target == wordIndex {
				continue
			}
			label = 0
		}

		targetVector := syn1neg[target]

		// Calculate dot product between hidden layer and target vector
		dot := float32(0.0)
		for i := range vectorSize {
			dot += neu1[i] * targetVector[i]
		}

		// Calculate prediction and error
		prediction := t.sigmoid(dot)
		gradient := alpha * (float32(label) - prediction)

		// Accumulate error for the hidden layer
		for i := range vectorSize {
			neu1e[i] += gradient * targetVector[i]
		}

		// Update the negative sampling weights
		for i := range vectorSize {
			targetVector[i] += gradient * neu1[i]
		}
	}

	// Backpropagate error to the context word vectors
	for _, contextWordIndex := range context {
		contextVector := syn0[contextWordIndex]
		for i := range vectorSize {
			contextVector[i] += neu1e[i]
		}
	}
}

// trainSkipGramHierarchicalSoftmax implements the Skip-gram model with Hierarchical Softmax.
func (t *Trainer) trainSkipGramHierarchicalSoftmax(wordIndex, contextWordIndex int, alpha float32, neu1e []float32) {
	t.model.mu.Lock()
	defer t.model.mu.Unlock()

	vectorSize := t.settings.VectorSize
	syn0 := t.model.syn0[wordIndex]
	syn1 := t.model.syn1

	// Clear the gradient buffer
	for i := range vectorSize {
		neu1e[i] = 0
	}

	huffmanPath := t.huffman.Points[contextWordIndex]
	huffmanCode := t.huffman.Codes[contextWordIndex]

	for i, point := range huffmanPath {
		pointVector := syn1[point]

		// Calculate dot product
		dot := float32(0.0)
		for j := 0; j < vectorSize; j++ {
			dot += syn0[j] * pointVector[j]
		}

		// Calculate prediction and error
		prediction := t.sigmoid(dot)
		label := float32(1 - huffmanCode[i]) // 1 if path is left (code 0), 0 if path is right (code 1)
		gradient := alpha * (label - prediction)

		// Accumulate error for the input word vector
		for j := 0; j < vectorSize; j++ {
			neu1e[j] += gradient * pointVector[j]
		}

		// Update the hidden layer (huffman node) vector
		for j := 0; j < vectorSize; j++ {
			pointVector[j] += gradient * syn0[j]
		}
	}

	// Update the input word's vector
	for i := range vectorSize {
		syn0[i] += neu1e[i]
	}
}

// trainCBOWHierarchicalSoftmax implements the CBOW model with Hierarchical Softmax.
func (t *Trainer) trainCBOWHierarchicalSoftmax(wordIndex int, context []int, alpha float32, neu1e []float32) {
	t.model.mu.Lock()
	defer t.model.mu.Unlock()

	vectorSize := t.settings.VectorSize
	syn0 := t.model.syn0
	syn1 := t.model.syn1
	neu1 := make([]float32, vectorSize) // Hidden layer activations

	// Sum the vectors of the context words
	for _, contextWordIndex := range context {
		for i := range vectorSize {
			neu1[i] += syn0[contextWordIndex][i]
		}
	}

	// Average the context vectors
	contextLen := float32(len(context))
	for i := range vectorSize {
		neu1[i] /= contextLen
	}

	huffmanPath := t.huffman.Points[wordIndex]
	huffmanCode := t.huffman.Codes[wordIndex]

	for i, point := range huffmanPath {
		pointVector := syn1[point]
		dot := float32(0.0)
		for j := 0; j < vectorSize; j++ {
			dot += neu1[j] * pointVector[j]
		}
		prediction := t.sigmoid(dot)
		label := float32(1 - huffmanCode[i])
		gradient := alpha * (label - prediction)

		// Backpropagate errors to context vectors
		for _, contextWordIndex := range context {
			for j := 0; j < vectorSize; j++ {
				syn0[contextWordIndex][j] += gradient * pointVector[j]
			}
		}

		// Update hidden layer weights
		for j := 0; j < vectorSize; j++ {
			pointVector[j] += gradient * neu1[j]
		}
	}
}
