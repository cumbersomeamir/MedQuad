#pip3 install transformers pandas openpyxl datasets sentencepiece

# Importing the relevant libraries
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, Trainer, LLaMATokenizer, LLaMAForCausalLM

# Reading the file
data = pd.read_excel("MedQuad dataset test.xlsx")

# Convert the pandas DataFrame to Hugging Face's Dataset
hf_dataset = Dataset.from_pandas(data)

# Tokenize the dataset
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Tokenization
def tokenize_function(examples):
    tokenized_prompt = tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    tokenized_completion = tokenizer(examples['completion'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    return {'input_ids': tokenized_prompt['input_ids'], 'attention_mask': tokenized_prompt['attention_mask'], 'labels': tokenized_completion['input_ids']}

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
print("The tokenized dataset is ", tokenized_dataset)

# Load the pre-trained GPT-Neo 1.3B model
model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-7b-hf')

# Define the training arguments
training_args = TrainingArguments(
    output_dir='output',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_total_limit=1,
    logging_steps=100,
    evaluation_strategy="no",
    num_gpus=torch.cuda.device_count(),  # Set the number of GPUs to use
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('finetuned_llama')

model = LlamaForCausalLM.from_pretrained('finetuned_llama')

# Saving the Model on huggingface
token = "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"
model.push_to_hub("Amirkid/finetune-llama-test", use_auth_token=token)
