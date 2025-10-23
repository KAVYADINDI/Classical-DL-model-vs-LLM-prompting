# -----------------------------------------------------------
# Script: generate_absa_dataset.py
# Purpose: Generate synthetic ABSA-style review dataset
# -----------------------------------------------------------

import pandas as pd
import random
from faker import Faker

fake = Faker()

# Define aspects & sentiment templates
aspects = [
    "Design", "Quality", "Durability", "Price",
    "Packaging", "Usability", "Material", "Experience"
]

sentiments = ["Positive", "Neutral", "Negative"]

positive_templates = [
    "The {aspect} of this {category} is excellent.",
    "Absolutely love the {aspect}, very well done!",
    "Superb {aspect}! It exceeded my expectations.",
    "Impressive {aspect}, highly recommend this {category}.",
    "The {aspect} gives a premium feel."
]

neutral_templates = [
    "The {aspect} is okay, nothing special.",
    "Average {aspect}, meets basic expectations.",
    "The {aspect} is decent for the price.",
    "Fair {aspect}, could be better.",
    "Not bad but not great {aspect}."
]

negative_templates = [
    "The {aspect} is disappointing.",
    "Poor {aspect}, not satisfied with this {category}.",
    "The {aspect} could use major improvement.",
    "Bad {aspect}, feels cheap.",
    "I expected better {aspect} quality."
]

# Example product list
products = [
    {"title": "Handmade Mug", "category": "Home Decor"},
    {"title": "Canvas Painting", "category": "Wall Art"},
    {"title": "Fabric Tote Bag", "category": "Wearable Art"},
    {"title": "Wooden Organizer", "category": "Utility Crafts"},
    {"title": "Greeting Card Set", "category": "Stationery"},
]

# Settings
records_per_product = 1000
data = []

for product in products:
    for _ in range(records_per_product):
        aspect = random.choice(aspects)
        sentiment = random.choice(sentiments)

        if sentiment == "Positive":
            text = random.choice(positive_templates)
            rating = random.randint(4, 5)
        elif sentiment == "Neutral":
            text = random.choice(neutral_templates)
            rating = random.randint(2, 4)
        else:
            text = random.choice(negative_templates)
            rating = random.randint(1, 3)

        review_text = text.format(aspect=aspect.lower(), category=product["category"].lower())

        data.append({
            "Product Title": product["title"],
            "Category": product["category"],
            "Aspect": aspect,
            "Review Text": review_text,
            "Sentiment": sentiment,
            "Rating": rating,
            "Reviewer Name": fake.first_name(),
            "Date": fake.date_between(start_date='-1y', end_date='today')
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Export to Excel
output_file = "absa_dataset.xlsx"
df.to_excel(output_file, index=False)

print(f"✅ ABSA dataset generated with {len(df)} records")
print(f"Saved to: {output_file}")
