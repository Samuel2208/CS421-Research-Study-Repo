# CS421 Research Study Group 2

## Authors

* Victor Escudero
* Shuroq Hussein
* Ibrahim Siddiqui

## Description

This repository was made for the research study done by group 2 for CS421 in Spring 2026 at UIC. We focused on LLMs ability to classify the emotion of an utterance based on previous context.

## Project Files

* Gemini_Tests
  * Located in this folder are the tests that use Google's Gemini LLM.
  * Tests include 2-Shot, Zero-Shot, and Least-to-Most prompting.
  * Each prompting method includes a structured and unstructured prompt test.
* gemma_tests
  * Located in this folder are tests that use Google's local LLM, Gemma 4.
  * Tests include 2-Shot, Zero-Shot, and Least-to-Most prompting.
  * Each prompting method includes a baseline prompt, a structured prompt, a lemmatized dialog prompt, and a structured and lemmatized dialog prompt test.
* qwen2_tests
  * Located in this folder are tests that use the local LLM Qwen2.5.
  * Tests include 2-Shot, Zero-Shot, and Least-to-Most prompting.
  * Each prompting method includes a baseline prompt, a structured prompt, a lemmatized dialog prompt, and a structured and lemmatized dialog prompt test.
* qwen3_tests
  * Note: The tests found in this folder are non-functional.
  * Located in this folder are tests that use the local LLM Qwen3.5.
  * Tests include 2-Shot, Zero-Shot, and Least-to-Most prompting.
  * Each prompting method includes a baseline prompt, a structured prompt, a lemmatized dialog prompt, and a structured and lemmatized dialog prompt test.
* Evaluators
  * Throughout the project there are `evaluating.py` or similar scripts in each test folder. This assumes that a .csv file is in the `results` folder for that test and it generates metrics based on the CSV.
* plotting
  * This folder contains plotting utilities to generate graphs based on the evaluated results.
* Tables
  * This folder contains various utilities to generate CSVs that have an overview of the results. It also contains graphing utilities for those new CSVs.
* utils
  * Contains an unused (it was used at some point but findings were not meaningful) lexicon analysis utility.
* Dataset
  * Contains the dataset files.
* Dataset_Code
  * Contains utilities that either generate filtered subsets of the datasets or various data for the datasets.

## Getting Started

### Dependencies
* Python 3.13.12 required
* Install dependencies in `requirements.txt` using pip.
  ```
  pip install -r requirements.txt
  ```

### Executing program

* Gemini Tests
  * 2-Shot & Zero-Shot
    * Create a `.env` file and add a variable named `GEMINI_API_KEY`.
    * Set `GEMINI_API_KEY` to your Gemini API key.
    * Run the test files.
  * Least-to-Most
    * In the line `client = genai.Client()`, within the parentheses put your Gemini API key.
    * Run the test files.
* Local LLM Tests (Qwen2 & Gemma)
  * In the root folder run `run_all_new.sh` to automatically run all local LLM tests.
  * Manually run all tests by navigating to the respective folder and running with Python.
  * Note: Qwen3.5 is in the repository, but is non-functional.

## Acknowledgments

* [DomPizzie's README-Template.md](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
  * The template used for this README.
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
  * Borrowed some elements for README items.
* S. Zahiri and J. D. Choi. Emotion Detection on TV Show Transcripts with Sequence-based Convolutional Neural Networks. In The AAAI Workshop on Affective Content Analysis, AFFCON'18, 2018.
* S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation. ACL 2019.
* T. Saha, A. Patra, S. Saha, P. Bhattacharyya. Towards Emotion-aided Multi-modal Dialogue Act Classification. ACL 2020.
* [MELD Dataset](https://github.com/sahatulika15/EMOTyDA/tree/master)
* [DailyDialog Dataset](https://www.kaggle.com/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in?select=train.csv)
