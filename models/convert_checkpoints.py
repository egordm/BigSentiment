#%%

from transformers import ElectraConfig, ElectraForPreTraining, load_tf_weights_in_electra, ElectraForMaskedLM
import os
import torch

#%%
MODEL_DIR = './data/models'

#%%
CONFIG_FILE = os.path.join(MODEL_DIR, 'discriminator/config.json')
config = ElectraConfig.from_pretrained(CONFIG_FILE)

print('Converting the discriminator model')
DUMP_PATH = os.path.join(MODEL_DIR, 'discriminator/pytorch_model.bin')
model = ElectraForPreTraining(config)
load_tf_weights_in_electra(model, config, os.path.join(MODEL_DIR, 'electra_bitcoin_twitternew'))

print('Saving the model')
torch.save(model.state_dict(), DUMP_PATH)

#%%
CONFIG_FILE = os.path.join(MODEL_DIR, 'generator/config.json')
config = ElectraConfig.from_pretrained(CONFIG_FILE)

print('Converting the generator model')
DUMP_PATH = os.path.join(MODEL_DIR, 'generator/pytorch_model.bin')
model = ElectraForMaskedLM(config)
load_tf_weights_in_electra(model, config, os.path.join(MODEL_DIR, 'electra_bitcoin_twitternew'))

print('Saving the model')
torch.save(model.state_dict(), DUMP_PATH)

#%%
