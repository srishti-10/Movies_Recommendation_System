import pandas as pd
import tez
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn import metrics,preprocessing
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieDataset:
    def __init__(self,users,movies,ratings):
        self.users=users
        self.movies=movies
        self.ratings=ratings

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self,item):
        user=self.users[item]
        movie=self.movies[item]
        rating=self.ratings[item]

        return {
            "users":torch.tensor(user,dtype=torch.long),
            "movies":torch.tensor(movie,dtype=torch.long),
            "ratings":torch.tensor(rating,dtype=torch.float)
              
        }



class RecSysModel(tez.Model):
    def __init__(self,num_users,num_movies):
        super().__init__()
        self.user_embed=nn.Embedding(num_users,32)
        self.movie_embed=nn.Embedding(num_movies,32)
        self.out=nn.Linear(64,1)
        self.step_scheduler_after='epoch'

    def fetch_optimizer(self):
        opt=torch.optim.Adam(self.parameters(),lr=1e-4)
        return opt
    
    def fetch_scheduler(self):
        sch= torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=3,gamma=0.7)
        return sch

    def monitor_metrics(self,output,rating):
        output=output.detach().cpu().numpy()
        rating=rating.detach().cpu().numpy()
        return {
            'rmse':np.sqrt(metrics.mean_squared_error(rating,output))
        }
        
    def forward(self,users,movies,ratings=None):
        user_embeds=self.user_embed(users)
        movie_embeds=self.movie_embed(movies)
        output= torch.cat([user_embeds,movie_embeds],dim=1)
        output=self.out(output)
        
        
        loss=nn.MSELoss()(output,ratings.view(-1,1))
        calc_metrics =self.monitor_metrics(output,ratings.view(-1,1))
        return output,loss,calc_metrics






def train():
    df= pd.read_csv('datasets/ratings_small.csv')
    lbl_user=preprocessing.LabelEncoder()
    lbl_movie=preprocessing.LabelEncoder()
    df.user=lbl_user.fit_transform(df.user.values)
    df.movie=lbl_movie.fit_transform(df.movie.values)

    df_train,df_valid=train_test_split(df,test_size=0.2,random_state=42,stratify=df.rating.values)
    train_dataset=MovieDataset(users=df_train.user.values,movies=df_train.movie.values,ratings=df_train.rating.values)
    valid_dataset=MovieDataset(users=df_valid.user.values,movies=df_valid.movie.values,ratings=df_valid.rating.values)
    model=RecSysModel(num_users=len(lbl_user.classes_), num_movies=len(lbl_movie.classes_))
    model.fit(
        train_dataset,valid_dataset,train_bs=1024,
        valid_bs=1024, fp16=True, device="cpu"
    )
    torch.save(model,'/models/recsysmodel.pth')

# --- Evaluation functions ---
def evaluate_model(model, valid_dataset):
    model.eval()
    actual = []
    preds = []
    with torch.no_grad():
        for i in range(len(valid_dataset)):
            batch = valid_dataset[i]
            users = batch['users'].unsqueeze(0)
            movies = batch['movies'].unsqueeze(0)
            ratings = batch['ratings'].unsqueeze(0)
            output, _, _ = model(users, movies, ratings)
            preds.append(output.item())
            actual.append(ratings.item())
    preds = np.array(preds)
    actual = np.array(actual)
    rmse = np.sqrt(metrics.mean_squared_error(actual, preds))
    print(f"Collaborative Filtering RMSE: {rmse:.4f}")
    # For recommendation precision/recall, see hybrid/content-based below

# --- Content-based recommender (modularized) ---
def evaluate_content_based(metadata_csv):
    df = pd.read_csv(metadata_csv)
    df = df.dropna(subset=['overview'])
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Example evaluation: recommend for a random movie
    idx = 0
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    recommended = [df.iloc[i[0]].title for i in sim_scores]
    print(f"Sample recommendations for '{df.iloc[idx].title}': {recommended}")
    # TODO: Add precision/recall evaluation using user history if available

# --- Hybrid recommender placeholder ---
def hybrid_recommend(user_id, movie_title, model, metadata_csv):
    # Combine collaborative and content-based for better results
    # TODO: Implement hybrid logic using user profile, embeddings, and content similarity
    pass

if __name__=="__main__":
    model, valid_dataset, df_valid = train()
    # Evaluate collaborative filtering model
    evaluate_model(model, valid_dataset)
    # Evaluate content-based recommender
    evaluate_content_based('movies_metadata.csv')
    # Ready for UI integration: main() or Streamlit app can call recommend functions