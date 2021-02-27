import os
# import wandb
from functools import partial

from datasets import load_dataset
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, \
    set_seed, DataCollatorForLanguageModeling, Trainer, ElectraTokenizerFast, \
    TrainingArguments, EvaluationStrategy
from transformers.tokenization_utils_base import PaddingStrategy
from models.ElectraPretrainingModel import ElectraPretrainingModel, DataCollatorForElectra

# wandb.login()
# wandb.init(project="bigbrain")

# Initial settings
tokenizer_custom = {
    '@HTAG': '[HTAG]',
    '@USR': '[USR]',
    '@CURR': '[CURR]',
    '@EMOJI': '[EMOJI]',
    '@URL': '[URL]',
    '@TIME': '[TIME]',
    '@DATE': '[DATE]',
    '@NUM': '[NUM]'
}
additional_tokens = list(tokenizer_custom.values())

DATASET_DIR = './data/bitcoin_twitter_corpus'
VOCAB_FILE = './data/bitcoin_twitter-vocab.txt'
TRAIN_DS = os.path.join(DATASET_DIR, 'train.tokens')
TEST_DS = os.path.join(DATASET_DIR, 'test.tokens')
LIGHT_DS = os.path.join(DATASET_DIR, 'light.tokens')
VALIDATE_DS = os.path.join(DATASET_DIR, 'validate.tokens')
DATASET_PRESAVE_DIR = './bitcoin_twitter_tokenized'

model_path = './bitcoin_twitter'
seq_length = 256
accum_multipler = 1
batch_size = 128
epochs = 1
warmup_ratio = 0.06
lr = 5e-4
vocab_size = 16537
block_size = 200
seed = 1337
train_batch_size = 4
eval_batch_size = 16

set_seed(seed)

# Load the curate vocabulary into the tokenizer
tokenizer = ElectraTokenizerFast(vocab_file=VOCAB_FILE)
tokenizer.add_special_tokens({
    'additional_special_tokens': list(tokenizer_custom.values())
})
assert tokenizer.vocab_size == vocab_size

# Load the dataset
dataset = load_dataset("text", data_files={
    'train': LIGHT_DS,
    # 'train': TRAIN_DS,
    # 'test': TEST_DS,
    # 'validate': VALIDATE_DS
}, cache_dir='./cache')

# Preprocess / Pre-tokenize the dataset
tokenize_function = partial(tokenizer, truncation=True, padding=PaddingStrategy.MAX_LENGTH, max_length=seq_length, return_token_type_ids=False)
tokenized_datasets = dataset.shuffle().map(lambda x: tokenize_function(x['text']), batched=True)
eval_dataset = tokenized_datasets['train'].train_test_split(test_size=100)['test']

# Build the model
generator_config = ElectraConfig(
    embedding_size=128,
    hidden_size=256,
    intermediate_size=1024,
    max_position_embeddings=seq_length,
    num_attention_heads=4,
    num_hidden_layers=12,
    vocab_size=vocab_size,
)

discriminator_config = ElectraConfig(
    embedding_size=128,
    hidden_size=256,
    intermediate_size=1024,
    max_position_embeddings=seq_length,
    num_attention_heads=4,
    num_hidden_layers=12,
    vocab_size=vocab_size,
)

generator = ElectraForMaskedLM(config=generator_config)
discriminator = ElectraForPreTraining(config=discriminator_config)
model = ElectraPretrainingModel(discriminator, generator, tokenizer, 'cpu')

# Setup training params
data_collator = DataCollatorForElectra(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

arguments = TrainingArguments(
    output_dir=model_path,
    do_train=True,
    evaluation_strategy=EvaluationStrategy.STEPS,
    eval_steps=10000,
    prediction_loss_only=True,
    learning_rate=lr,
    load_best_model_at_end=True,
    num_train_epochs=20,
    # report_to=['wandb'],
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    run_name='pretrain_electra',
)

# Initialize our Trainer
trainer = Trainer(
    args=arguments,
    model=model,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    # eval_dataset=tokenized_datasets['validate'],
)

# Training
trainer.train()
trainer.save_model()
