# Word2vec in Go

This is a native Go implementation of the word2vec algorithm, inspired by the original C implementation. The project has been completely rewritten in pure Go, which eliminates the need for CGo and external C dependencies. This makes it easy to build and use in your own Go projects.

## Features

* **Pure Go**: No CGo or external system dependencies. All you need is `go get`.
* **Two Model Architectures**: Supports **Skip-gram** and **Continuous Bag of Words (CBOW)**.
* **Two Training Algorithms**: Supports **Negative Sampling** and **Hierarchical Softmax**.
* **Command-Line Interface (CLI)**: A powerful tool for training models, finding semantically similar words, and calculating embeddings for documents.
* **Library Usage**: A simple and clear API for integrating word2vec into your own Go applications.

## Installation

To use the CLI tool, you can clone the repository and build it:

```bash
git clone https://github.com/kirill-scherba/word2vec.git
cd word2vec
go build -o w2v ./cmd/w2v
```

This command will create an executable file named `w2v` in the project root.

To use it as a library in your project:

```bash
go get github.com/kirill-scherba/word2vec
```

## Usage

The project provides both a CLI tool and a library.

### Command-Line Interface

The `w2v` executable has three main subcommands: `train`, `distance`, and `embedding`. You can get help for any command by adding the `--help` flag.

#### 1. Train a model (`train`)

Use this command to train a new model on your text corpus.

```bash
./w2v train --train text8 --output vectors.bin --size 200 --window 8 --negative 5 --threads 12
```

## License

[BSD](LICENSE)
