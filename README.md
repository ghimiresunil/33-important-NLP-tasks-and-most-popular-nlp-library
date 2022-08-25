# 33 important NLP tasks explained

NLP landscape with a brief explanation of 33 common NLP tasks. I will try to make it as simple as I can.

**1. Classification**
- Text Classification: assigning a category to a sentence or document (e.g. spam filtering).
- Sentiment Analysis: identifying the polarity of a piece of text.

**2. Information Retrieval and Document Ranking**
- Information Retrieval and Document Ranking
- Question Answering: the task of answering a question in natural language.

**3. Text-to-Text Generation**
- Machine Translation: translating from one language to another.
- Text Generation: creating text that appears indistinguishable from human-written text.
- Text Summarization: creating a shortened version of several documents that preserves most of their meaning.
- Text Simplification: making a text easier to read and understand, while preserving its main ideas and approximate meaning.
- Lexical Normalization: translating/transforming a non-standard text to a standard register.
- Paraphrase Generation: creating an output sentence that preserves the meaning of input but includes variations in word choice and grammar.

**4. Knowledge bases, entities and relations**
- Relation extraction: extracting semantic relationships from a text. Extracted relationships usually occur between two or more entities and fall into specific semantic categories (e.g. lives in, sister of, etc).
- Relation prediction: identifying a named relation between two named semantic entities.
- Named Entity Recognition: tagging entities in text with their corresponding type, typically in BIO notation.
- Entity Linking: recognizing and disambiguating named entities to a knowledge base (typically Wikidata).

**5. Topics and Keywords**
- Topic Modeling: identifying abstract “topics” underlying a collection of documents.
- Keyword Extraction: identifying the most relevant terms to describe the subject of a document

**6. Chatbots**
- Intent Detection: capturing the semantics behind messages from users and assigning them to the correct label.
- Slot Filling: aims to extract the values of certain types of attributes (or slots, such as cities or dates) for a given entity from texts.
- Dialog Management: managing of state and flow of conversations.

**7. Text Reasoning**
- Common Sense Reasoning: use of “common sense” or world knowledge to make inferences.
- Natural Language Inference: determining whether a “hypothesis” is true (entailment), false (contradiction), or undetermined (neutral) given a “premise”.

**8. Fake News and Hate Speech Detection**
- Fake News Detection: detecting and filtering out texts containing false and misleading information.
- Stance Detection: determining an individual’s reaction to a primary actor’s claim. It is a core part of a set of approaches to fake news assessment.
- Hate Speech Detection: detecting if a piece of text contains hate speech.

**9. Text-to-Data and viceversa**
- Text-to-Speech: technology that reads digital text aloud.
- Speech-to-Text: transcribing speech to text.
- Text-to-Image: generating photo-realistic images which are semantically consistent with the text descriptions.
- Data-to-Text: producing text from non-linguistic input, such as databases of records, spreadsheets, and expert system knowledge bases.

**10. Text Preprocessing**
- Coreference Resolution: clustering mentions in text that refer to the same underlying real-world entities.
- Part Of Speech (POS) tagging: tagging a word in a text with its part of speech. A part of speech is a category of words with similar grammatical properties, such as noun, verb, adjective, adverb, pronoun, preposition, conjunction, etc.
- Word Sense Disambiguation: associating words in context with their most suitable entry in a pre-defined sense inventory (typically WordNet).
- Grammatical Error Correction: correcting different kinds of errors in text such as spelling, punctuation, grammatical, and word choice errors.
- Feature Extraction: extraction of generic numerical features from text, usually embeddings.

# Awesome 20 popular NLP libraries of 2022

Here is the list of top libraries, sorted by their number of GitHub stars.

**1. [Hugging Face Transformers](https://github.com/huggingface/transformers)**
- 69k GitHub stars
- Transformers provides thousands of pre-trained models to perform tasks on different modalities such as text, vision, and audio. These models can be applied to text (text classification, information extraction, question answering, summarization, translation, text generation, in over 100 languages), images (image classification, object detection, and segmentation), and audio (speech recognition and audio classification). Transformer models can also perform tasks on several modalities combined, such as table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.
-  Currently updated.

**2. [spaCy](https://github.com/explosion/spaCy)**
- 24k GitHub stars.
- spaCy is a free open-source library for Natural Language Processing in Python and Cython. It’s built on the very latest research and was designed from day one to be used in production environments. spaCy comes with pre-trained pipelines and currently supports tokenization and training for 60+ languages. It features state-of-the-art speed and neural network models for tagging, parsing, named entity recognition, text classification, multi-task learning with pre-trained transformers like BERT, as well as a production-ready training system and easy model packaging, deployment, and workflow management. spaCy is commercial open-source software, released under the MIT license.
- Currently updated.

**3. [Fairseq](https://github.com/facebookresearch/fairseq)**
- 19k GitHub stars
- Fairseq is a sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling, and other text generation tasks. It provides reference implementations of various sequence modeling papers.
- Currently updated.

**4. [Jina](https://github.com/jina-ai/jina)**
- 15.8k GitHub stars
- Jina is a neural search framework to build state-of-the-art and scalable neural search applications in minutes. Jina allows building solutions for indexing, querying, understanding multi-/cross-modal data such as video, image, text, audio, source code, PDF.
- Currently updated

**5. [Gensim](https://github.com/RaRe-Technologies/gensim)**
- 13.5k GitHub stars.
- Gensim is a Python library for topic modeling, document indexing, and similarity retrieval with large corpora. The target audience is the NLP and information retrieval (IR) community. Gensim has efficient multicore implementations of popular algorithms, such as online Latent Semantic Analysis (LSA/LSI/SVD), Latent Dirichlet Allocation (LDA), Random Projections (RP), Hierarchical Dirichlet Process (HDP), or word2vec deep learning.
- Currently updated.

**6. [Flair](https://github.com/flairNLP/flair)**
- 12k GitHub stars.
- Flair is a powerful NLP library. Flair allows you to apply state-of-the-art NLP models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS), special support for biomedical data, sense disambiguation and classification, with support for a rapidly growing number of languages. Flair has simple interfaces that allow you to use and combine different word and document embeddings, including Flair embeddings, BERT embeddings, and ELMo embeddings. The framework builds directly on PyTorch, making it easy to train your own models and experiment with new approaches using Flair embeddings and classes.
- Currently updated.

**7. [AllenNLP](https://github.com/allenai/allennlp)**
- 11.2k Github stars.
- An Apache 2.0 NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks. It provides a broad collection of existing model implementations that are well documented and engineered to a high standard, making them a great foundation for further research. AllenNLP offers a high-level configuration language to implement many common approaches in NLP, such as transformer experiments, multi-task training, vision+language tasks, fairness, and interpretability. This allows experimentation on a broad range of tasks purely through configuration, so you can focus on the important questions in your research.
- Currently updated.

**8. [NLTK](https://github.com/nltk/nltk)**
- 11k Github stars
- NLTK — the Natural Language Toolkit — is a suite of open-source Python modules, data sets, and tutorials supporting research and development in Natural Language Processing. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries.
- Currently updated.

**9. [CoreNLP](https://github.com/stanfordnlp/CoreNLP)**
- 8.6k Github stars
- Stanford CoreNLP provides a set of natural language analysis tools written in Java. It can take raw human language text input and give the base forms of words, their parts of speech, whether they are names of companies, people, etc., normalize and interpret dates, times, and numeric quantities, mark up the structure of sentences in terms of phrases or word dependencies, and indicate which noun phrases refer to the same entities.
- Currently updated.

**10. [Pattern](https://github.com/clips/pattern)**
- 8.3k Github stars
- Pattern is a web mining module for Python. It has tools for data mining: web services (Google, Twitter, Wikipedia), web crawler, and HTML DOM parser. It has several Natural Language Processing models: part-of-speech taggers, n-gram search, sentiment analysis, and WordNet. It implements Machine Learning models: vector space model, clustering, classification (KNN, SVM, Perceptron). Pattern can be also used for Network Analysis: graph centrality and visualization.
- Last update 4 years ago.

**11. [TextBlob](https://github.com/sloria/TextBlob)**
- 8.3k Github stars
- TextBlob is a Python library for processing textual data. It provides a simple API for diving into common Natural Language Processing tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more. TextBlob stands on the giant shoulders of NLTK and Pattern and plays nicely with both.
- Currently updated.

**12. [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)**
- 5.8k GitHub stars
- This library provides an implementation of today’s most used tokenizers, with a focus on performance and versatility.
- Currently updated.

**13. [Haystack](https://github.com/deepset-ai/haystack/)**
- 3.8k GitHub stars.
- Haystack is an end-to-end framework that enables you to build powerful and production-ready pipelines for different search use cases. Whether you want to perform Question Answering or semantic document search, you can use the State-of-the-Art NLP models in Haystack to provide unique search experiences and allow your users to query in natural language. Haystack is built in a modular fashion so that you can combine the best technology from other open-source projects like Huggingface’s Transformers, Elasticsearch, or Milvus.
- Currently updated.

**14. [Snips NLU](https://github.com/snipsco/snips-nlu)**
- 3.7k GitHub stars.
- Snips NLU is a Python library that allows the extraction of structured information from sentences written in natural language. Anytime a user interacts with an AI using natural language, their words need to be translated into a machine-readable description of what they meant. The NLU (Natural Language Understanding) engine of Snips NLU first detects what the intention of the user is (a.k.a. the intent), then extracts the parameters (called slots) of the query.
- Last update 3 years ago.

**15. [NLP Architect](https://github.com/IntelLabs/nlp-architect)**
- 2.9k GitHub stars.
- NLP Architect is an open-source Python library for exploring state-of-the-art deep learning topologies and techniques for optimizing Natural Language Processing and Natural Language Understanding Neural Networks. It’s a library designed to be flexible, easy to extend, allowing for easy and rapid integration of NLP models in applications, and to showcase optimized models.
- Currently updated

**16. [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP)**
- 2.1k GitHub stars.
- PyTorch-NLP is a library of basic utilities for PyTorch NLP. It extends PyTorch to provide you with basic text data processing functions.
- Currently updated.

**17. [Polyglot](https://github.com/aboSamoor/polyglot)**
- 2k GitHub stars.
- Polyglot is a natural language pipeline that supports massive multilingual applications: Tokenization (165 Languages), Language Detection (196 Languages), Named Entity Recognition (40 Languages), Part of Speech Tagging (16 Languages), Sentiment Analysis (136 Languages), Word Embeddings (137 Languages), Morphological analysis (135 Languages), and Transliteration (69 Languages).
- Last updated 2 years ago

**18. [TextAttack](https://github.com/QData/TextAttack)**
- 2.1k GitHub stars.
- TextAttack is a Python framework for adversarial attacks, data augmentation, and model training in NLP.
- Currently updated.

**19. [Word Forms](https://github.com/gutfeeling/word_forms)**
- 549 GitHub stars.
- Word forms can accurately generate all possible forms of an English word. It can conjugate verbs and pluralize singular nouns. It can connect different parts of speeches e.g noun to adjective, adjective to adverb, noun to verb, etc.
- Last update 1 year ago.

**20. [Rosetta](https://github.com/LatticeX-Foundation/Rosetta)**
-  496 GitHub stars.
- Rosetta is a privacy-preserving framework based on TensorFlow. It integrates with mainstream privacy-preserving computation technologies, including cryptography, federated learning, and trusted execution environment. Rosetta reuses the APIs of TensorFlow and allows the transfer of traditional TensorFlow codes into a privacy-preserving manner with minimal changes.
- Currently updated.
