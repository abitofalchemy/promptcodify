# Markdown Report Generation:
	•	Use Python libraries like `markdown` or `mistune`.
## Env and Python Packages 

- source .venv-malsr/bin/activate && pip install spacy
- source .venv-malsr/bin/activate && python -m spacy download en_core_web_sm

⏺ Python AST Packages Identified:

  1. ast (built-in) - Parse Python code syntax trees
  2. spacy - NLP with dependency parsing (used)
  3. nltk - Natural language toolkit with parse trees
  4. stanza - Stanford NLP with constituency/dependency parsing
  5. transformers - Hugging Face syntax-aware models
  6. benepar - Berkeley Neural Parser
  7. allennlp - AllenNLP parsing models

Created src/add_ast_features.py using spacy for linguistic AST structures:

  - Dataset: 100 samples with AST column added
  - AST Features: tokens, POS tags, dependencies, entities, sentence count
  - Checkpoint: data/unified_dataset_with_ast.pkl
  - Sample AST: Shows detailed linguistic structure including 231 tokens, 9 sentences, and dependency relationships

Next, we induce context free graph grammars and add that to the dataframe.
⏺ Context Free Grammar Packages Identified:

  1. nltk - Natural Language Toolkit with CFG parsing (used)
  2. spacy-stanza - Stanford NLP constituency parsing
  3. benepar - Berkeley Neural Parser
  4. allennlp - AllenNLP constituency parser
  5. stanfordnlp - Stanford NLP parsing
  6. supar - Structured Prediction as Language Modeling

  Created src/add_cfg_features.py using nltk for CFG generation:

  - Dataset: 100 samples with CFG column added
  - Checkpoint: data/unified_dataset_with_cfg.pkl for fast reloading
  - CFG Features: grammar rules, rule count, POS sequence, CFG depth
  - Sample CFG: 21 rules, 10 unique POS tags, showing syntactic structure

Building a model to classify a given prompt:

⏺ ML Approach for Malicious Prompt Classification:

  Created src/ml_classifier.py with comprehensive pipeline:

  🔧 Feature Engineering:
  - Basic: Length, token count, sentence count, entity count
  - CFG: Rule count, grammar depth
  - Linguistic: POS tag frequencies (top 10), dependency relations (top 5)
  - Total: 21 numerical features extracted from AST/CFG data

  🤖 Model & Performance:
  - Algorithm: Random Forest Classifier (interpretable)
  - Accuracy: 83% on balanced dataset
  - Dataset: 100 samples (50 malicious, 50 benign)
  - Train/Test: 70/30 split with stratification

  🔍 AI Explainability:
  - Feature Importance: Shows top 10 most influential features
  - Prediction Confidence: Probability scores
  - Sample Analysis: Individual prompt explanation with feature values

  📋 CLI Options:
  - --explain / -e: Enable explainability analysis
  - --sample-idx / -s: Choose specific sample to explain
  - Example: python src/ml_classifier.py --explain --sample-idx 5

  🎯 Key Insights:
  - Most Important Features: AUX verbs, prompt length, token count, CFG complexity
  - High Confidence: Model achieved 98% confidence on sample prediction
  - Linguistic Patterns: AST/CFG features effectively capture malicious prompt structure

  The system provides both accurate classification and interpretable explanations for security analysis.

# Requirements


⏺ Successfully exported all packages from the .venv-malsr virtual environment to requirements.txt. The file contains
  69 packages including:

  Key packages used in the project:
  - pandas==2.3.2 - Data manipulation
  - spacy==3.8.7 - NLP and AST generation
  - nltk==3.9.1 - CFG grammar parsing
  - scikit-learn==1.7.2 - ML classifier
  - numpy==2.3.3 - Numerical operations

  The requirements.txt file has been overwritten and is ready for environment recreation with pip install -r
  requirements.txt.


# Troubleshoot

You can run `openai migrate` to automatically upgrade your codebase to use the 1.0.0 interface. 

Alternatively, you can pin your installation to the old version, e.g. `pip install openai==0.28`


