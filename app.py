import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
@st.cache_data
def load_data():
    books_df = pd.read_csv('/Users/salomonmuhirwa/Desktop/book r system/books/Books.csv', dtype={'Year-Of-Publication': str})
    users_df = pd.read_csv('/Users/salomonmuhirwa/Desktop/book r system/books/Users.csv')
    with open('user_asset_interaction_matrix.pkl', 'rb') as f:
        user_asset_matrix = pickle.load(f)
    return books_df, users_df, user_asset_matrix

@st.cache_data
def clean_data(books_df, users_df):
    books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
    books_df = books_df.dropna(subset=['Year-Of-Publication'])
    users_df = users_df.dropna(subset=['Location'])
    return books_df, users_df

@st.cache_data
def calculate_cosine_similarity(user_asset_matrix):
    cosine_sim = cosine_similarity(user_asset_matrix.fillna(0))
    cosine_sim_df = pd.DataFrame(cosine_sim, index=user_asset_matrix.index, columns=user_asset_matrix.index)
    return cosine_sim_df

def get_nearest_neighbors(user_id, similarity_df, k=5):
    neighbors = similarity_df.loc[user_id].sort_values(ascending=False).head(k).index
    return neighbors

def generate_recommendations(user_id, nearest_neighbors, interaction_matrix):
    neighbor_interactions = interaction_matrix.loc[nearest_neighbors].sum(axis=0)
    target_user_interactions = interaction_matrix.loc[user_id]
    recommendations = neighbor_interactions[target_user_interactions == 0].sort_values(ascending=False).index
    return recommendations

def display_recommendations(user_id, recommendations, asset_data):
    recommended_assets = asset_data[asset_data['ISBN'].isin(recommendations)]
    return recommended_assets[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]

def add_new_user(users_df, user_asset_matrix, new_user):
    new_user_df = pd.DataFrame([new_user])
    users_df = pd.concat([users_df, new_user_df], ignore_index=True)
    users_df['User-ID'] = users_df['User-ID'].astype(int)  # Ensure User-ID is int
    new_user_id = new_user['User-ID']
    if new_user_id not in user_asset_matrix.index:
        user_asset_matrix.loc[new_user_id] = 0
    return users_df, user_asset_matrix

def save_interaction_matrix(user_asset_matrix):
    with open('user_asset_interaction_matrix.pkl', 'wb') as f:
        pickle.dump(user_asset_matrix, f)
st.title('Recommendation System')

# Load and clean data
books_df, users_df, user_asset_matrix = load_data()
books_df, users_df = clean_data(books_df, users_df)

# New user registration
st.header('Register New User')
new_user_id = st.number_input('Enter New User ID', min_value=1, step=1)
new_user_location = st.text_input('Enter Location')
new_user_submit = st.button('Register User')

if new_user_submit:
    new_user = {'User-ID': new_user_id, 'Location': new_user_location}
    users_df, user_asset_matrix = add_new_user(users_df, user_asset_matrix, new_user)
    users_df.to_csv('/Users/salomonmuhirwa/Desktop/book r system/books/Users.csv', index=False)
    save_interaction_matrix(user_asset_matrix)
    st.write(f"User ID {new_user_id} registered successfully.")
    
    # Debugging: Print out the users_df and user_asset_matrix to verify new user
    st.write("Updated Users DataFrame:")
    # st.write(users_df.tail())
    st.write("Updated User-Asset Interaction Matrix:")
    # st.write(user_asset_matrix.tail())

# Calculate cosine similarity
cosine_sim_df = calculate_cosine_similarity(user_asset_matrix)
# st.write("Cosine Similarity Matrix:")
#st.write(cosine_sim_df)

# Input user ID for recommendations
st.header('Get Recommendations')
user_id = st.number_input('Enter User ID for Recommendations', min_value=1, step=1)

if st.button('Get Recommendations'):
    if user_id in users_df['User-ID'].values:
        if user_id in cosine_sim_df.index:
            nearest_neighbors = get_nearest_neighbors(user_id, cosine_sim_df)
            recommendations = generate_recommendations(user_id, nearest_neighbors, user_asset_matrix)
            recommended_assets = display_recommendations(user_id, recommendations, books_df)

            if not recommended_assets.empty:
                st.write(f"Recommendations for User ID {user_id}:")
                for _, row in recommended_assets.iterrows():
                    st.image(row['Image-URL-L'], caption=f"{row['Book-Title']} by {row['Book-Author']} ({row['Year-Of-Publication']})")
            else:
                st.write(f"No recommendations available for User ID {user_id}.")
        else:
            st.write(f"User ID {user_id} is not in the cosine similarity matrix.")
    else:
        st.write(f"No user found with the ID {user_id}.")
        st.write("Current Users DataFrame:")
        st.write(users_df)
