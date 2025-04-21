from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Create a small movie dataset
movies_data = {
    'movie_id': range(1, 21),
    'movie_title': [
        'the dark knight', 'inception', 'interstellar', 'pulp fiction',
        'the matrix', 'fight club', 'forrest gump', 'the godfather',
        'goodfellas', 'the shawshank redemption', 'avatar', 'titanic',
        'jurassic park', 'the lion king', 'star wars', 'gladiator',
        'the avengers', 'the silence of the lambs', 'saving private ryan',
        'the lord of the rings'
    ],
    'genres': [
        'action crime drama', 'action sci-fi thriller', 'adventure drama sci-fi',
        'crime drama', 'action sci-fi', 'drama', 'drama romance',
        'crime drama', 'biography crime drama', 'drama',
        'action adventure fantasy', 'drama romance', 'adventure sci-fi',
        'animation drama', 'action adventure fantasy', 'action drama',
        'action sci-fi', 'crime drama thriller', 'drama war',
        'action adventure fantasy'
    ]
}

# Convert to DataFrame
movies_df = pd.DataFrame(movies_data)

# Create feature vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_df['genres']).toarray()
similarity = cosine_similarity(vectors)

@app.route("/")
def home():
    return {"message": "Movie recommendation system is running"}

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        movie_name = data['movie_name'].strip().lower()
        
        # Find the movie in our dataset
        movie_index = movies_df[movies_df['movie_title'].str.contains(movie_name, case=False)].index
        
        if len(movie_index) == 0:
            return jsonify({"error": "Movie not found"}), 404
            
        movie_index = movie_index[0]
        similar_movies = list(enumerate(similarity[movie_index]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]
        
        recommendations = []
        for i in sorted_similar_movies:
            recommendations.append(movies_df['movie_title'].iloc[i[0]].title())
            
        return jsonify({"recommendations": recommendations})
    
    except Exception as e:
        app.logger.error(f'An error occurred: {e}')
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

