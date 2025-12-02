from simpletransformers.ner import NERModel, NERArgs
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Example training data
train_data = [
    [0, "Alice", "B-PER"],
    [0, "works", "O"],
    [0, "at", "O"],
    [0, "OpenAI", "B-ORG"],
    [0, ".", "O"],
    [1, "Bob", "B-PER"],
    [1, "lives", "O"],
    [1, "in", "O"],
    [1, "Paris", "B-LOC"],
    [1, ".", "O"],
]

train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])

# Example evaluation data
eval_data = [
    [2, "Charlie", "B-PER"],
    [2, "is", "O"],
    [2, "from", "O"],
    [2, "Google", "B-ORG"],
    [2, ".", "O"],
]

eval_df = pd.DataFrame(eval_data, columns=["sentence_id", "words", "labels"])

# Model arguments
model_args = NERArgs()
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.num_train_epochs = 3
model_args.train_batch_size = 16

# Initialize the NER model using CPU (my laptop has a very weak gpu)
model = NERModel(
    "bert",
    "bert-base-cased",
    args=model_args,
    use_cuda=False
)

# Train the model
model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
result, model_outputs, predictions = model.eval_model(eval_df)
print("Evaluation result:", result)

preds, raw_outputs = model.predict(["John Doe works at Microsoft in London ."])
print("Predictions:", preds)
