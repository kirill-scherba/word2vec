# Word2vec in Go

This is a native Go implementation of the word2vec algorithm, inspired by the original C implementation. The project has been completely rewritten in pure Go, which eliminates the need for CGo and external C dependencies. This makes it easy to build and use in your own Go projects.

## Features

* **Pure Go**: No CGo or external system dependencies.
* **Two Model Architectures**: Supports **Skip-gram** and **Continuous Bag of Words (CBOW)**.
* **Two Training Algorithms**: Supports **Negative Sampling** and **Hierarchical Softmax**.
* **Optional Stemming**: Built-in Porter stemmer for Russian, which can be enabled with a flag during training.
* **Command-Line Interface (CLI)**: A powerful tool for training models, finding semantic similarities, and performing analogy tasks.
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

The `w2v` executable has five main subcommands: `train`, `distance`, `embedding`, `analogy`, and `similarity`. You can get help for any command by adding the `--help` flag.

#### 1. Train a model (`train`)

Use this command to train a new model on your text corpus.

**Basic training:**

```bash
./w2v train --train <path_to_text_corpus> --output vectors.bin
```

**Advanced training with Russian stemming:**

```bash
./w2v train --train <russian_text.txt> --output vectors_stemmed.bin --lemmatize
```

**Key flags:**

* `--train`: Path to the training text file (required).
* `--output`: File to save the resulting vectors.
* `--size`: Set vector dimensions (default 100).
* `--window`: Set max skip length between words (default 5).
* `--lemmatize`: Enable Russian stemming during training. This is useful for grouping different forms of a word.
* `--min-count`: Discard words that appear less than `<int>` times (default 5).
* `--threads`: Number of CPU threads to use (default 12).
* `--iter`: Number of training epochs (default 5).

#### 2. Find nearest words (`distance`)

After training a model, you can find semantically similar words.

```bash
./w2v distance --model vectors.bin --word <your_word>
```

#### 3. Calculate an embedding (`embedding`)

This command calculates the vector representation (embedding) for an entire sentence or document by averaging the vectors of its words.

```bash
./w2v embedding --model vectors.bin --text "some text to embed"
```

#### 4. Perform an analogy task (`analogy`)

This is the classic test for word2vec. It computes `word1 - word2 + word3`. For this to work well, the model must be trained on a very large and general text corpus (e.g., Wikipedia).

```bash
./w2v analogy --model <large_model.bin> --words король,мужчина,женщина
```

### Library Usage

You can easily integrate `word2vec` into your Go applications.

**Example:**

```go
package main
import (
    "fmt"
    "log"

    "github.com/kirill-scherba/word2vec"
)

func main() {
    // Load the model
    model, err := word2vec.Load("vectors.bin")
    if err != nil {
        log.Fatalf("failed to load model: %v", err)
    }

    // Find the 10 nearest words to a query word
    queryWord := "example"
    nearestWords := make([]word2vec.Nearest, 10)
    if err := model.Lookup(queryWord, nearestWords); err != nil {
        log.Fatalf("lookup failed: %v", err)
    }

    fmt.Printf("Words closest to '%s':\n", queryWord)
    for _, word := range nearestWords {
        fmt.Printf("%20s\t%f\n", word.Word, word.Distance)
    }
}
```

## License

[BSD](LICENSE)
