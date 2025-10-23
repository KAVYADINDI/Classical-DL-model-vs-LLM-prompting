# -----------------------------------------------------
# Script: export_product_reviews.py
# Purpose: Extract product reviews from MongoDB Atlas 
#          and export them to Excel for AI analysis.
# -----------------------------------------------------

import pandas as pd
from pymongo import MongoClient

# 1. MongoDB connection setup
# Replace <username>, <password>, and <cluster-url> with your actual Atlas details
MONGO_URI = "mongodb+srv://<username>:<password>@<cluster-url>/yourDatabaseName?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)

# 2. Select database and collection
db = client["yourDatabaseName"]
collection = db["products"]  # Your collection name

# 3. Extract product data
data = []
cursor = collection.find({}, {"title": 1, "category": 1, "reviews": 1, "rating": 1, "_id": 0})

for doc in cursor:
    product_title = doc.get("title")
    category = doc.get("category")
    overall_rating = doc.get("rating", 0)
    reviews = doc.get("reviews", [])

    # Each review entry
    for review in reviews:
        data.append({
            "Product Title": product_title,
            "Category": category,
            "Overall Rating": overall_rating,
            "Reviewer Name": review.get("name"),
            "Review Rating": review.get("rating"),
            "Review Comment": review.get("comment"),
            "User ID": str(review.get("user", "")),
            "Review Created": review.get("createdAt", "")
        })

# 4. Convert to DataFrame
df = pd.DataFrame(data)

# Optional: clean missing data
df.fillna("", inplace=True)

# 5. Export to Excel
output_file = "product_reviews_dataset.xlsx"
df.to_excel(output_file, index=False)

print(f"Export completed! File saved as: {output_file}")
print(f"Total reviews exported: {len(df)}")