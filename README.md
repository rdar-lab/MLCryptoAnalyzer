![](RackMultipart20200717-4-ixlw2n_html_22fbd9ac23f4e463.gif) ![](RackMultipart20200717-4-ixlw2n_html_a826b0333c6d8886.gif) ![](RackMultipart20200717-4-ixlw2n_html_9ecae252f34899a7.gif)

# ML Cypto Analyzer

by Roy Dar


## Introduction

One definition of a learning algorithm is its ability to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E (Tom Mitchell/1998).

In the last couple of years, the ML field has improved significantly allowing sophisticated tasks to be performed using learning algorithms.

A significant part of the improvement in this field is attributed to the increase in computational power allowing models which were only theoretical in the past to become practical (e.g. neural networks).

In this study, I attempted to find out if a learning algorithm can leak information about a cipher text.

In this trial I tried both classic/simple encryption methods (for example &quot;Caesar&quot; like cipher or simple XOR cipher) and more modern ciphers (e.g. AES, DES, 3-DES).

The trials I used had the following goals:

1. Detecting the encryption schema that was used
2. Detecting the type of the original plaintext (e.g. binary file, English text)

## Project description

The project implementation was done in **python 3.x**.

The following libraries were used:

1. **numpy** – mathematic library for python
2. **pycryptodome** – An implementation of modern ciphers for python
3. **tensorflow** – A base core component for ML implementation
4. **keras** – A more advanced API for ML on top of tensorflow to easily create, train and test more advanced and powerful models.
5. **termplot** – A small library allowing plotting charts on terminal

For more details see the MLCryptoAnalyzer.pdf in the project

## The trials (tests)

All test sizes are currently tweaked to run under the machine requirements (see running section below). To run on other machines the sizes of input sizes, batch sizes and network sizes can be changed in code. On the tests below the aim is to minimize loss (needs to be as close to 0) and maximize the accuracy (as close to 1).

All the trials use on-line training data generation. Originally, I generated a file stored training set and used it, but I got into scenarios of over fitting to the training data. I then switched to on-line generation and found it to have no implication on the performance.

For all trials I used a mix of the types and sizes of NN and the types of input encoding.

Due to computational resource limitations, I was limited in the sizes of inputs and model complexity I could use.

The results you see before you are the ones that gave me the best performance.

The project includes the following tests. Each can be run separately.

The list of trials is laid out on the next pages.

### Trial #1 – A sanity test

This test aim was to see if the NN is performing as expected under a very easy task.

The ML in this task receives an un-encrypted plaintext, and its job is just to learn the classification between a dictionary (English) text or binary data.

Model: FC NN, 1 hidden layer

Training: 50 rounds, each round of 100 training example

Plaintext size: Variable size up to 2000 bytes

Training time: a few seconds

Result: Success


### Trial #2 – Detect plaintext type from a legacy SHIFT encrypted ciphertext

In this trial I used the SHIFT encryption schema with a long but hard-coded (single) key.

The job of the ML was that, given enough classified inputs, recognize the type of plaintext from the cipher text (without knowing the encryption key of course)

Model: FC NN, 1 hidden layer

Training: 50 rounds, each round of 100 training example

Plaintext size: Variable size up to 2000 bytes

Training time: a few seconds

Result: Success



### Trial #3 – Detect plaintext type from a legacy XOR encrypted ciphertext

In this trial I used the XOR encryption schema with a long but hard-coded (single) key.

The job of the ML was that, given enough classified inputs, recognize the type of plaintext from the cipher text (without knowing the encryption key of course)

Model: FC NN, 1 hidden layer

Training: 50 rounds, each round of 100 training example

Plaintext size: Variable size up to 2000 bytes

Training time: a few seconds

Result: Success



### Trial #4 - Detect plaintext type from real OTP (one-time pad) encrypted ciphertext

In this trial I used XOR but with random key for each training example. As proven and expected, the ML failed to recognize the plain text type

Model: FC NN, 1 hidden layer

Training: 50 rounds, each round of 100 training example

Plaintext size: Variable size up to 2000 bytes

Training time: a few seconds

Result: Success



### Trial #5 – Detect the type of cipher that was used between XOR or SHIFT

In this trial I used the same plaintext type (English text) but this time the encryption method changed (XOR, SHIFT). I used the same key for each one (as we saw above if the key randomly changes for each training example, ML will be unable to detect anything.

The ML job was given the ciphertext detect the type of encryption schema that was used.

Model: FC NN, 5 hidden layers – 100 units per layer

Training: 50 rounds, each round of 100 training example

Plaintext size: Variable size up to 2000 bytes

Training time: a few seconds

Result: Success



### Trial #6 - Detect plaintext type from AES (single key)

Moving on to modern ciphers I now used AES with a single key to encrypt all the training examples. The job of the ML was that, given enough classified inputs, recognize the type of plaintext from the cipher text.

To my surprise the ML model was unable to detect the types even when I used a single key for all the training examples and even after 1000 rounds of training on freshly randomized data. I tried FC NN, CNN and RNN networks but those all failed.

Model: FC NN, 5 hidden layers – 100 units per layer

Training: 1000 rounds, each round of 100 training example

Plaintext size: Variable size up to 2000 bytes

Training time: a few minutes

Result: Failure



### Trial #7 – Same as #6 only with DES encryption

After the failure to detect anything from AES ciphertext I tried the same for DES. Surprising enough, also here the ML failed to detect anything as well.

Model: FC NN, 5 hidden layers – 100 units per layer

Training: 1000 rounds, each round of 100 training example

Plaintext size: Variable size up to 2000 bytes

Training time: a few minutes

Result: Failure



### Trial #8 – Detect the type of cipher that was used between AES or DES

This trial is similar to #5, only using AES and DES (both using a single key for all training examples)

Again, the plaintext is English text and the ML goal is to recognize which cipher was used AES or DES.

The result surprised me because the ML was able, to a degree higher than randomness, detect some differences between the cipher texts that were generated.

Model: FC NN, 5 hidden layers – 100 units per layer

Training: 1000 rounds, each round of 1000 training example

Plaintext size: Variable size up to 2000 bytes (with embedding as one-hot enabled)

Training time: a few minutes

Result: Partial Success (not perfect but \&gt;80% of accuracy)



### Trial #9 – Detect the type of cipher that was used between AES or DES (changing keys)

Same as #8 only now the keys are random and changes between each training example.

As expected, the ML failed to detect the cipher that was used.

Model: FC NN, 5 hidden layers – 100 units per layer

Training: 1000 rounds, each round of 100 training example

Plaintext size: Variable size up to 2000 bytes

Training time: a few minutes

Result: Failure


### Trial #10 – Same as #2 only with random changing keys of small size

Using SHIFT encryption schema with a short random key.

The job of the ML was that, given enough classified inputs, recognize the type of plaintext from the cipher text.

Model: FC NN, 5 hidden layers – 100 units per layer

Training: 100 rounds, each round of 100 training example

Plaintext size: Variable size up to 2000 bytes

Training time: a few minutes

Result: Success



### Trial #11 – Same as #2 only with RNN (with variable sizes of inputs).

The AIM of this test, was to test the performance of RNN, in using variable sizes of inputs, from very small to very large (100 – 10,000)

As in trial #2, I was using a simple SHIFT cipher with fixed large key.

Model: RNN network (Conv1D -\&gt; LSTM -\&gt; LSTM -\&gt; FC -\&gt; ACTIVATION) with dropout ratio 0.05 between each layer

Training: 100 rounds, each round of 100 training example

Plaintext size: Variable size from 100 to 10,000 bytes

Training time: a few minutes

Result: Success on all sizes


## Running this project

Recommended machine specs for running this project:

1. CPU – Intel i7 8th generation or above
2. Memory – 32 Gb DDRM
3. Disk - ~2GB of free disk space
4. OS: Project was developed using Windows 10 64bit, but should be Linux compatible
5. GPU – Nvidia 1060 GPU with 6GB of mem with CUDA installed and configured correctly for ML projects (Running the project without this can result in 10-100X slower training)

To install and run this project:

1. Install python 3.x on your machine (tested with version 3.7.1)
2. Install pip on your machine
3. Clone or download the sources
4. Go to the installation directory
5. Execute &quot;pip install -r requirements.txt&quot; to install all the project library requirements. Here is the list of the required libraries
  - numpy (tested with version 1.18.4)
  - pycryptodome (tested with version 3.9.7)
  - keras (tested with version 2.3.1)
  - tensorflow (tested with version 2.2.0)
  - termplot (tested with version 0.0.2)
6. Download the English dictionary for this project
  - For the English text generation this project requires the dictionary file named &quot;glove.6B.50d.txt&quot;.
  - The dictionary needs to be downloaded and saved into the /resources directory
  - The dictionary can be downloaded from the following location: https://www.kaggle.com/watts2/glove6b50dtxt](https://www.kaggle.com/watts2/glove6b50dtxt)
7. Execute &quot;python main.py&quot; to run the project
8. You will be prompted with a menu allowing you the select the trial you wish to run
