// Copyright 2026 Kirill Scherba <kirill@scherba.ru>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libw2v

import "container/heap"

// HuffmanNode represents a node in the Huffman tree.
// It can be either a leaf (a word) or a branch (an internal node).
type huffmanNode interface {
	getCount() int64
}

// huffmanLeaf represents a word in the vocabulary.
type huffmanLeaf struct {
	count     int64
	wordIndex int
}

func (l huffmanLeaf) getCount() int64 { return l.count }

// huffmanBranch represents an internal node connecting two other nodes.
type huffmanBranch struct {
	count int64
	left  huffmanNode
	right huffmanNode
}

func (b huffmanBranch) getCount() int64 { return b.count }

// nodeHeap is a min-heap of Huffman nodes, used as a priority queue.
type nodeHeap []huffmanNode

func (h nodeHeap) Len() int           { return len(h) }
func (h nodeHeap) Less(i, j int) bool { return h[i].getCount() < h[j].getCount() }
func (h nodeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *nodeHeap) Push(x any) {
	*h = append(*h, x.(huffmanNode))
}

func (h *nodeHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// HuffmanTree holds the generated codes and paths for each word.
type HuffmanTree struct {
	// Codes stores the Huffman code for each word.
	Codes [][]byte
	// Points stores the path from the root to each word.
	Points [][]int
}

// NewHuffmanTree builds a Huffman tree from the vocabulary and generates codes.
func NewHuffmanTree(vocab *Vocabulary) *HuffmanTree {
	vocabSize := vocab.Len()
	if vocabSize == 0 {
		return &HuffmanTree{}
	}

	// 1. Initialize the priority queue with leaf nodes from the vocabulary.
	pq := make(nodeHeap, vocabSize)
	for i, word := range vocab.Words() {
		pq[i] = huffmanLeaf{count: word.Count, wordIndex: i}
	}
	heap.Init(&pq)

	// 2. Build the tree by merging nodes.
	for pq.Len() > 1 {
		left := heap.Pop(&pq).(huffmanNode)
		right := heap.Pop(&pq).(huffmanNode)

		newBranch := huffmanBranch{
			count: left.getCount() + right.getCount(),
			left:  left,
			right: right,
		}
		heap.Push(&pq, newBranch)
	}

	root := heap.Pop(&pq).(huffmanNode)

	// 3. Traverse the tree to build codes and points for each word.
	tree := &HuffmanTree{
		Codes:  make([][]byte, vocabSize),
		Points: make([][]int, vocabSize),
	}
	buildCodes(tree, root, []byte{}, []int{})

	return tree
}

// buildCodes is a recursive helper to traverse the tree.
func buildCodes(tree *HuffmanTree, node huffmanNode, code []byte, point []int) {
	switch n := node.(type) {
	case huffmanLeaf:
		// A leaf is reached, store the generated code and path.
		tree.Codes[n.wordIndex] = make([]byte, len(code))
		copy(tree.Codes[n.wordIndex], code)
		tree.Points[n.wordIndex] = make([]int, len(point))
		copy(tree.Points[n.wordIndex], point)

	case huffmanBranch:
		// A branch is reached, continue traversing.
		// The 'point' array stores indices of parent nodes.
		// The 'code' array stores the path (0 for left, 1 for right).
		newPoint := append(point, 0) // Placeholder, will be updated later
		buildCodes(tree, n.left, append(code, 0), newPoint)
		buildCodes(tree, n.right, append(code, 1), newPoint)

	default:
		// This should not happen
		panic("invalid node type in huffman tree")
	}
}
