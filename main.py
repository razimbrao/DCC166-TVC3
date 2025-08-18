import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
import gc
from typing import List, Tuple, Dict
import pickle
import os

warnings.filterwarnings('ignore')

class OptimizedMovieRecommender:

    def __init__(self, n_factors: int = 50, min_user_ratings: int = 5, min_movie_ratings: int = 10):

        self.n_factors = n_factors
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings
        
        self.user_factors = None
        self.item_factors = None
        self.user_means = None
        self.global_mean = None
        self.movie_popularity = None
        
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        
        self.movies_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        
    def load_data(self, movies_path: str, ratings_path: str) -> None:

        print("Loading data...")
        
        self.movies_df = pd.read_csv(movies_path, dtype={'movieId': 'int32'})
        print(f"Movies loaded: {len(self.movies_df):,}")
        
        chunk_list = []
        chunk_size = 50000
        
        for chunk in pd.read_csv(ratings_path, chunksize=chunk_size, 
                                 dtype={'userId': 'int32', 'movieId': 'int32', 
                                        'rating': 'float32', 'timestamp': 'int64'}):
            chunk_list.append(chunk)
        
        ratings_df = pd.concat(chunk_list, ignore_index=True)
        print(f"Ratings loaded: {len(ratings_df):,}")
        
        del chunk_list
        gc.collect()
        
        self.ratings_df = ratings_df
        
    def preprocess_data(self) -> None:
        print("\nStarting preprocessing...")
        
        initial_ratings = len(self.ratings_df)
        self.ratings_df.drop_duplicates(subset=['userId', 'movieId'], inplace=True)
        print(f"Duplicates removed: {initial_ratings - len(self.ratings_df):,}")
        
        user_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_ratings].index
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(valid_users)]
        print(f"Users filtered (min {self.min_user_ratings} ratings): {len(valid_users):,}")
        
        movie_counts = self.ratings_df['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= self.min_movie_ratings].index
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(valid_movies)]
        print(f"Movies filtered (min {self.min_movie_ratings} ratings): {len(valid_movies):,}")
        
        self.movies_df = self.movies_df[self.movies_df['movieId'].isin(valid_movies)]
        
        unique_users = sorted(self.ratings_df['userId'].unique())
        unique_movies = sorted(self.ratings_df['movieId'].unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.idx_to_movie = {idx: movie for movie, idx in self.movie_to_idx.items()}
        
        self.ratings_df['user_idx'] = self.ratings_df['userId'].map(self.user_to_idx)
        self.ratings_df['movie_idx'] = self.ratings_df['movieId'].map(self.movie_to_idx)
        
        print(f"Final data: {len(unique_users):,} users, {len(unique_movies):,} movies")
        print(f"Sparsity: {(1 - len(self.ratings_df) / (len(unique_users) * len(unique_movies))) * 100:.2f}%")
        
        del user_counts, movie_counts, valid_users, valid_movies
        gc.collect()

    def _calculate_popularity(self):
        print("\nCalculating movie popularity...")
        
        movie_counts = self.ratings_df.groupby('movieId').size().reset_index(name='n_ratings')
        
        movie_counts['log_ratings'] = np.log1p(movie_counts['n_ratings'])
        
        scaler = MinMaxScaler(feature_range=(0.5, 5.0))
        movie_counts['popularity_score'] = scaler.fit_transform(movie_counts[['log_ratings']])
        
        self.movie_popularity = movie_counts.set_index('movieId')['popularity_score'].to_dict()
        
        print(f"Popularity scores calculated for {len(self.movie_popularity):,} movies.")

    def create_user_item_matrix(self) -> None:
        print("\nCreating user-item matrix...")
        
        n_users = len(self.user_to_idx)
        n_movies = len(self.movie_to_idx)
        
        self.user_item_matrix = csr_matrix(
            (self.ratings_df['rating'].values, 
             (self.ratings_df['user_idx'].values, self.ratings_df['movie_idx'].values)),
            shape=(n_users, n_movies)
        )
        
        print(f"Matrix created: {n_users:,} x {n_movies:,}")
        print(f"Matrix memory: {self.user_item_matrix.data.nbytes / 1024**2:.2f} MB")
        
    def apply_svd(self) -> None:
        print(f"\nApplying SVD with {self.n_factors} factors...")
        
        user_ratings_sum = np.array(self.user_item_matrix.sum(axis=1)).flatten()
        user_ratings_count = np.array((self.user_item_matrix != 0).sum(axis=1)).flatten()
        
        self.user_means = np.divide(user_ratings_sum, user_ratings_count, 
                                     out=np.zeros_like(user_ratings_sum, dtype=float), 
                                     where=user_ratings_count!=0)

        self.global_mean = self.user_item_matrix.data.mean()
        
        matrix_centered = self.user_item_matrix.copy().astype(np.float32)
        for i in range(matrix_centered.shape[0]):
            user_ratings = matrix_centered.getrow(i)
            if user_ratings.nnz > 0:
                user_ratings.data -= self.user_means[i]
        
        print("Executing SVD decomposition...")
        U, sigma, Vt = svds(matrix_centered, k=self.n_factors, random_state=42)
        
        self.user_factors = U
        self.item_factors = Vt.T
        
        print(f"SVD completed!")
        print(f"User factors: {self.user_factors.shape}")
        print(f"Item factors: {self.item_factors.shape}")
        
        del matrix_centered
        gc.collect()
        
    def calculate_user_similarity(self, user_id: int, top_k: int = 50) -> List[Tuple[int, float]]:
        if user_id not in self.user_to_idx:
            return []
            
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_factors[user_idx].reshape(1, -1)
        
        similarities = cosine_similarity(user_vector, self.user_factors)[0]
        
        similar_indices = similarities.argsort()[::-1][1:top_k+1]
        similar_users = [(self.idx_to_user[idx], similarities[idx]) 
                         for idx in similar_indices if similarities[idx] > 0]
        
        return similar_users
        
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return self.global_mean
            
        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
        prediction += self.user_means[user_idx]
        
        return max(0.5, min(5.0, prediction))
        
    def recommend_movies(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        if user_id not in self.user_to_idx:
            print(f"User {user_id} not found in the system")
            return []
            
        print(f"\nGenerating recommendations for user {user_id}...")
        
        user_movies = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
        
        similar_users = self.calculate_user_similarity(user_id, top_k=20)
        print(f"Similar users found: {len(similar_users)}")
        
        candidate_movies = set()
        similar_user_ids = [uid for uid, _ in similar_users[:10]]
        
        for similar_user_id in similar_user_ids:
            similar_user_movies = set(self.ratings_df[
                (self.ratings_df['userId'] == similar_user_id) & 
                (self.ratings_df['rating'] >= 3.0)
            ]['movieId'])
            candidate_movies.update(similar_user_movies - user_movies)
        
        if len(candidate_movies) < n_recommendations * 5:
            popular_movies = self.ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            popular_movies.columns = ['movieId', 'avg_rating', 'count']
            popular_movies = popular_movies[
                (popular_movies['count'] >= 50) & 
                (popular_movies['avg_rating'] >= 4.0)
            ].sort_values('count', ascending=False)
            
            additional_candidates = set(popular_movies['movieId'].head(200)) - user_movies
            candidate_movies.update(additional_candidates)
        
        candidate_movies = list(candidate_movies)[:min(1000, len(candidate_movies))]
        print(f"Optimized candidate movies: {len(candidate_movies):,}")
        
        similar_ratings = {}
        for similar_user_id in similar_user_ids:
            user_ratings = self.ratings_df[self.ratings_df['userId'] == similar_user_id]
            similar_ratings[similar_user_id] = dict(zip(user_ratings['movieId'], user_ratings['rating']))
        
        movie_scores = {}
        print("Calculating scores...")
        
        for i, movie_id in enumerate(candidate_movies):
            if i % 200 == 0 and i > 0:
                print(f"   Processed: {i}/{len(candidate_movies)}")
                
            svd_score = self.predict_rating(user_id, movie_id)
            
            collaborative_score = 0
            weight_sum = 0
            
            for similar_user_id, similarity in similar_users[:10]:
                if similar_user_id in similar_ratings and movie_id in similar_ratings[similar_user_id]:
                    rating = similar_ratings[similar_user_id][movie_id]
                    collaborative_score += similarity * rating
                    weight_sum += similarity
            
            if weight_sum > 0:
                collaborative_score /= weight_sum
            else:
                collaborative_score = self.global_mean
            
            popularity_score = self.movie_popularity.get(movie_id, self.global_mean)

            final_score = (0.6 * svd_score) + \
                          (0.2 * collaborative_score) + \
                          (0.2 * popularity_score)
                          
            movie_scores[movie_id] = final_score
        
        top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        recommendations = []
        for movie_id, score in top_movies:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            recommendations.append({
                'movieId': movie_id,
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'predicted_rating': round(score, 2),
                'svd_score': round(self.predict_rating(user_id, movie_id), 2)
            })
        
        return recommendations
        
    def evaluate_model(self, test_size: float = 0.2) -> Dict[str, float]:
        print(f"\nEvaluating model (test_size={test_size})...")
        
        train_data, test_data = train_test_split(
            self.ratings_df[['userId', 'movieId', 'rating']], 
            test_size=test_size, 
            random_state=42
        )
        
        predictions = []
        actuals = []
        
        print("Calculating predictions...")
        for _, row in test_data.sample(min(1000, len(test_data)), random_state=42).iterrows():
            pred = self.predict_rating(row['userId'], row['movieId'])
            predictions.append(pred)
            actuals.append(row['rating'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        metrics = {
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'samples_evaluated': len(predictions)
        }
        
        print(f"Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value}")
            
        return metrics
        
    def save_model(self, filepath: str) -> None:
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_means': self.user_means,
            'global_mean': self.global_mean,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'movie_to_idx': self.movie_to_idx,
            'idx_to_movie': self.idx_to_movie,
            'n_factors': self.n_factors,
            'movie_popularity': self.movie_popularity
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to: {filepath}")
        
    def load_model(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        for key, value in model_data.items():
            setattr(self, key, value)
        print(f"Model loaded from: {filepath}")

def main():
    MOVIES_PATH = "datasets/movies.csv"
    RATINGS_PATH = "datasets/ratings.csv"
    
    print("Starting Movie Recommender System")
    print("=" * 60)
    
    recommender = OptimizedMovieRecommender(
        n_factors=50,
        min_user_ratings=5,
        min_movie_ratings=10
    )
    
    recommender.load_data(MOVIES_PATH, RATINGS_PATH)
    recommender.preprocess_data()
    
    recommender._calculate_popularity()
    
    recommender.create_user_item_matrix()
    recommender.apply_svd()
    
    metrics = recommender.evaluate_model()
    
    recommender.save_model("movie_recommender_model.pkl")
    
    print("\n" + "=" * 60)
    print("SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    sample_users = list(recommender.user_to_idx.keys())[:3]
    
    for user_id in sample_users:
        print(f"\nUSER {user_id}")
        print("-" * 40)
        
        user_history = recommender.ratings_df[recommender.ratings_df['userId'] == user_id]
        print(f"History: {len(user_history)} movies rated")
        print(f"Average rating: {user_history['rating'].mean():.2f}")
        
        recommendations = recommender.recommend_movies(user_id, n_recommendations=5)
        
        print(f"\nTOP 5 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   Genres: {rec['genres']}")
            print(f"   Prediction: {rec['predicted_rating']}/5.0")
            print()
    
    print("Recommender System finished successfully!")

if __name__ == "__main__":
    main()