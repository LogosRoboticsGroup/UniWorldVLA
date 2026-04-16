import pickle
import random

with open("dataset/navsim/scene_index.pkl", "rb") as f:
    scene_index = pickle.load(f)

sample_items = random.sample(scene_index.items(),10)

for k,v in sample_items:
    print(f'{k}: {v}')
