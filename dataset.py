import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
books_df = pd.read_csv('/Users/salomonmuhirwa/Desktop/book r system/books/Books.csv', dtype={'Year-Of-Publication': str})
users_df = pd.read_csv('/Users/salomonmuhirwa/Desktop/book r system/books/Users.csv')

# Clean up data
books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')

# Filter out incomplete or irrelevant data
books_df = books_df.dropna(subset=['Year-Of-Publication'])
users_df = users_df.dropna(subset=['Location'])

# Create user-asset interaction matrix
nyc_users = users_df[users_df['Location'].str.contains('nyc', case=False, na=False)]
recent_books = books_df[books_df['Year-Of-Publication'] > 2000]

user_asset_matrix = pd.DataFrame(index=nyc_users['User-ID'], columns=recent_books['ISBN'])

for user_id, user_row in nyc_users.iterrows():
    for book_id, book_row in recent_books.iterrows():
        if 'machine learning' in book_row['Book-Title'].lower():
            user_asset_matrix.loc[user_id, book_row['ISBN']] = 1
        else:
            user_asset_matrix.loc[user_id, book_row['ISBN']] = 0

user_asset_matrix = user_asset_matrix.fillna(0)
print(user_asset_matrix)
# cosine_sim = cosine_similarity(user_asset_matrix)
# cosine_sim_df = pd.DataFrame(cosine_sim, index=user_asset_matrix.index, columns=user_asset_matrix.index)

# def get_nearest_neighbors(user_id, similarity_df, k=5):
#     neighbors = similarity_df[user_id].sort_values(ascending=False).head(k).index
#     return neighbors

# def generate_recommendations(user_id, nearest_neighbors, interaction_matrix):
#     neighbor_interactions = interaction_matrix.loc[nearest_neighbors].sum(axis=0)
#     target_user_interactions = interaction_matrix.loc[user_id]
#     recommendations = neighbor_interactions[target_user_interactions == 0].sort_values(ascending=False).index
#     return recommendations

# def display_recommendations(user_id, recommendations, asset_data):
#     recommended_assets = asset_data[asset_data['ISBN'].isin(recommendations)]
#     return recommended_assets[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]

# # Streamlit App
# st.title('Book Recommendation System')

# user_id = st.number_input('Enter User ID', min_value=1, step=1)

# if st.button('Get Recommendations'):
#     if user_id in nyc_users['User-ID'].values:
#         nearest_neighbors = get_nearest_neighbors(user_id, cosine_sim_df)
#         recommendations = generate_recommendations(user_id, nearest_neighbors, user_asset_matrix)
#         recommended_assets = display_recommendations(user_id, recommendations, books_df)
        
#         if not recommended_assets.empty:
#             st.write(f"Recommendations for User ID {user_id}:")
#             for _, row in recommended_assets.iterrows():
#                 st.image(row['Image-URL-L'], caption=f"{row['Book-Title']} by {row['Book-Author']} ({row['Year-Of-Publication']})")
#         else:
#             st.write(f"No recommendations available for User ID {user_id}.")
#     else:
#         st.write(f"No user found with the ID {user_id}.")
