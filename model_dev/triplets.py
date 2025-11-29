import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

start = time.time()

raw_data_path = r"data\spectogrammes"
actors = sorted(os.listdir(raw_data_path))

# --- Split actors en Train / Test ---
train_actors, test_actors = train_test_split(
    actors, test_size=0.2, random_state=42
)

print("Train actors:", len(train_actors))
print("Test actors :", len(test_actors))

def load_images(actor_list):
    images_by_actor = {}
    for actor in actor_list:
        actor_dir = os.path.join(raw_data_path, actor)
        images = [
            os.path.join(actor_dir, f) 
            for f in os.listdir(actor_dir) 
            if f.endswith(".png")
        ]
        if len(images) > 1:
            images_by_actor[actor] = images
    return images_by_actor


train_images_by_actor = load_images(train_actors)
test_images_by_actor = load_images(test_actors)


def generate_triplets(images_by_actor, all_actors):
    triplets = []
    for actor, images in images_by_actor.items():
        other_actors = [a for a in all_actors if a != actor]
        max_pairs = min(600, len(images) * len(images))
        for _ in range(max_pairs):
            anchor, positive = np.random.choice(images, 2, replace=False)
            negative_actor = np.random.choice(other_actors)
            negative = np.random.choice(images_by_actor[negative_actor])
            triplets.append((anchor, positive, negative))
    
    return triplets


print("\nGenerating TRAIN triplets...")
train_triplets = generate_triplets(train_images_by_actor, train_actors)

print("Generating TEST triplets...")
test_triplets = generate_triplets(test_images_by_actor, test_actors)

df_train = pd.DataFrame(train_triplets, columns=["anchor", "positive", "negative"])
df_train.to_csv("data/triplet_dataset/train_triplets.csv", index=False)

df_test = pd.DataFrame(test_triplets, columns=["anchor", "positive", "negative"])
df_test.to_csv("data/triplet_dataset/test_triplets.csv", index=False)


end = time.time()
print(f"\nTime taken: {end - start:.2f} seconds")
