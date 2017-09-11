
#  Anime_Rec
##  A Data Science Portfolio Project by Sam Kenkel

Purpose: Build a recommendation engine to recommend Anime (Japanese Cartoons) based on the existing data on MyAnimeList.net.

Anime fans (or 'Otaku', as they prefer to be called), like consumers of any media, can use assistance from modern machine learning and data science techniques in deciding Anime to watch.

The project consist of three phases, Data Collection (webscraping large quantities of user reviews and  score from myanimelist.net), item-item filtering(using user scores, tf-idf of user reviews, and latent factors to return anime that are similar to anime user's like), and score prediction (5 different neural nets who are trained on differently processed data).


## The Data
#### Collection Process
A number of scripts are used to gather data,  because MyAnimeList (referred to as mal from here on out) has a history of DDOS attacks, they no longer give out production API keys.
As such, all data for this project was acquired via scraping.

##### Usernames
MAL does not give any way to enumerate users, except via a search function.
MAL_username Scrape generates a pseudo random collection of valid mal usernames by randomly choosing a three digit alphanumeric string, searching for usernames containing that string, getting every user, and repeating.
I stopped collecting usernames once I had 1.5 million user_names, as I deemed that to be a large enough sample to allow for adequate data collection.

##### User Scores
Once I have a set of usernames (A list->set-> list conversion removes duplicates nicely) Mal-Score-Scraper is used.
This script uses Mal-Scraper (https://github.com/QasimK/mal-scraper ) to get scores (a number from 1-10) and status (Watching, Completed, On Hold, etc) for each user.  Once I have this information I use pandas to save it to a dataframe in memory, then write out to a csv.
It's important to note that I parallelized this aspect of the scrape (with three systems using a vpn and different vpn endpoints) and that this could be easily parallelized to an arbitrary number of systems.



The raw data of user scores looks like:
![/readmepics/Screenshot%20from%202017-07-25%2009-01-00.png](/readmepics/Screenshot%20from%202017-07-25%2009-01-00.png?raw=true)

After running a scraper, I can convert this to a data frame:

![dataframe.png](/readmepics/dataframe.png?raw=true)

Anime_Score_Scraper.ipynb shows the code I used for this.
It's important to note that I have a UNKEY (Which is unique key) field. This allows for multiple score files to be merged later and duplicates easily removed.


```python

```

### Reviews
Users can give reviews of shows:
![review.png](/readmepics/review.png?raw=true)

We can scrape into a dataframe
![reviewframe.png](/readmepics/reviewframe.png?raw=true)

This gives me plaintext reviews of 6000 out of 11000 shows (the other 5000 shows lack reviews because they are too obscure for a user to have actively written a review. This is another example of nonresponse bias in this dataset).  
Anime_Review_scraper is used for this purpose. To prep for this, another tool (Anime_Info_Scraper) had to pull the URLs of every Anime's review page. (While each Anime's info page can be found by using myanimelist.net/ANIME_ID/
to get to an anime's review page  you must have the correct myanimelist.net/ANIME_ID/Anime_NAME/Reviews address.
Anime_Info_Scrape was used to generate that url, get a a description and genre for every anime, and get the url of the user-recommendations pages (for future use).



# EDA
### Exploratory Data analysis
We can look at this data, and notice some important trends:

Here are scores with the 0's removed.
![scoredistribution.png](/readmepics/scoredistribution.png?raw=true)

Here is a comparison of average scores based on status.
![score%20by%20status.png](/readmepics/score_by_status.png?raw=true)

AVG Scores a user gives as a function of how many reviews the user has done.
![reviews%20per%20user.png](/readmepics/reviews%20per%20user.png?raw=true)

Notably:
There are significant differences in how otaku score shows that they have not seen, but plan to vs shows they  are seeing vs shows  have seen and decided to stop watching.

In the Data, any time a user indicates they are watching, will watch or have begun watching but stopped a show without giving it a score, the score is represented by a 0.



# Data (Continued)
### Other Notes:
It's important to note that all of this data is explicit user feedback from a site that requires users to register in order to rate shows.

This means that the "Average" user of this site is likely not a casual fan of Anime (The 'nonresponse' bias of fans who are not willing to register on a website is significant).  Any analysis, or modeling done on this data may not apply to casual Anime fans.



Finally, as with all explicit feedback based data on media 'Guilty Pleasures' are not modeled well. Anime that is popular and controversial, but that Otaku do not rate highly in a public forum, will not be well served.

The data  does not have any time elements to it, so separating scores that users have given to shows recently, or shows that they cannot have seen (because they have not yet aired) or shows that they likely have seen years ago is impossible with the data available.

# Architecture
As a brief explanation of the overall architecture,
Once the Score and Review scrapers have been used to generate full data-lake of user scores and Anime Reviews, there is multi-step process used to generate a recommendation.  To show the larger architecture before going into the components:

![anime_rec_arch3.jpg](/readmepics/anime_rec_arch3.jpg?raw=true)

In short, Each user's N highest scored shows (scaled by the user's avg score, and scaled by the Anime's avg score) are used to make seed shows.

In a 'production' environment, if the user had rated less than N these would be determined by multi-arm bandit testing, or by business needs.

Each of these seed shows is fed to three different item-item similarity filters, which return the K most similar Anime (ignoring Anime that the user has either scores or marked a status for.

This generates between K (if the K most similar non-completed anime from each of the three filters were the same, for every show) and 3KN (if there is no overlap) of 'possible recommendations'.
In a production environment new releases or a static list of shows could be inserted at this step.

The 'possible recommendations' are all fed to 3 different user score prediction  systems (Neural Nets).
All of these neural nets have the same architecture, but are trained to predict a different kind of score.
Each of these networks predicts the user's score for each of  'possible recommendations', and returns a ranked list.

Those T-3T Anime are the recommendations for the user.

# Generating Possible Recommendations
### Three Types of Item-Item Similarity filtering

##### Type 1: User Score Cosine Distance collaborative filtering
Anime_Score_Sim shows the code for this.
First, we drop all 0's.
Next, we find the average score for each user, and subtract that score from every user's score. This is normalize the data (Without doing this, user's having different ideas of what constitutes a 6 vs a 7 vs 8, adds a lot of noise to the data).
Then make a sparse matrix of every user x every anime (with the user's scaled scores being at the intersection).
After some remapping to get those into anime names for eda, we can spot check to confirm that our similar anime are logical:
![users_cosim_deathnote.png](\readmepics\users_cosim_deathnote.png?raw=true)
If we look at Death Note, one of the most popular and successful anime of all time, we get a number of other very popular and successful shows, including Attack On Titan (which is directed by the same director, Tetsuro Araki). However, it's hard to judge these recommendations as this almost just a list of 'Crossover Anime'.

Let's look at something more obscure:
![user_score_cosim_girlsundpanzer.png](\readmepics\user_score_cosim_girlsundpanzer.png?raw=true)
Girls Und Panzer, the twee show about 'Cute Girls riding in Tanks as a highschool sport' is a little more informative. We see a great number of spin-off and sequel works (MAL lists prequels and sequels as different anime), as well as a number of other 'Cute Girls doing Cute Things shows', such as Hidamari Sketch  and Non Non Biyori.

If we examine the polarizing show Bakemonogatari:
![user_score_cosrim_bakemono.png](\readmepics\user_score_cosrim_bakemono.png?raw=true)
We see one of the downsides of this form of recommendation system.  We see 10 recommendations for the sequels to Bakemonogatari, and 1 recommendation for another work by the same author (Katanagatari). While this certainly indicates that these are similar shows, it is likely that using this system of similarity with the MAL dataset will give 'possible recs' that the user has already seen, but not scored, or is already aware of.

##### Type 2: Anime Review  NLP TF-IDF   Cosine Distance collaborative filtering
Review_Sim shows the code for this.
We begin by merging all of our reviews  into a one dataframe and removing duplicates.
Then we generate a dataframe that consists of each row containing one  anime with every review's text merged together.
Then we use a TF-IDF (Term Frequency, Inverse Document Frequency) vectorizer to generate a sparse matrix for each anime. We use some custom stop-words (such as anime), which we want the vectorizer to ignore.
N-grams are set from 1-3, which means that two word phrases (Such as 'moe trash', are treated as a word to vectorized and counted)
In short, this is taking every 1 to 3 word phrase in every review, determining how often that 'word' appears in review text for one anime (such as cowboy bebop) and compares it to how often that 'word' appears in all reviews.

There are few ways we can vectorize this, namely, we can do all words, with all counts, all words with a binary(count of 0 or 1) or we can slice this down to 900 most common words.
To clean up the text before we vectorize it, we remove some common stop words, and replace that name of the anime with the word 'anime_name' (which is also ignored)
With this sparse matrix of values, we can once again compute cosine pairwise distance between all shows. (So shows which are described by anime fans using similar uncommon words will be closer to each other).
We then turn these similarity scores into a dataframe which allows us to look at shows most similar to each other.

Then we check the results.
For Death Note we see:
![tfidf3deathnote.png](\readmepics\tfidf3deathnote.png?raw=true)
A who's who of very popular, well liked anime, but this is a different list than user score distance gave us. (Specifically, this list tilts towards older, 'darker' shows.

If we check Girls Und Panzer:
![tfidf3girlsundpanzer.png](\readmepics\tfidf3girlsundpanzer.png?raw=true)
we see excellent results. This list is not only different from the user score distances, but is a wide variety of "Moe" (Cute girls doing cute things) shows, with the highest scored show being "High School Fleet" which is a show that has the same premise, but with battleships rather than tanks.

When we look at Bakemonogatari,
![tfidf3bakemonogatari.png](\readmepics\tfidf3bakemonogatari.png?raw=true)
we also see a marked improvement. We see the shows direct sequel (Nisemonogatari), and a show based on a novel by the same author (Katanagatari), but we see a number of similar shows (Durara, Spice and Wolf, Mawaru Penguindrum) that are different from what user score distance returned.

##### Type 3: Aime Hidden Factors( HF)    Cosine Distance collaborative filtering
Mal_Keras_Fm_Clustering shows the code for this.
What I do is make a factorization machine neural net. This neural net embeds each user and each anime as N hidden factors, then makes the score a simple dot product of these hidden factors.
After training this neural network, we use the embedding for each show (which are the shows 'hidden factors') and do cosine pairwise distance.
By changing the number of hidden factors, and the amount of regularization, we can modify how 'hidden' these factors become.

Here is the network that is used to generate this

![dot2.png](/readmepics/dot2.png?raw=true)

The more Latent factors and the heavier the regularization, the stranger the similar anime become. Without regularization and too few hidden factors the results overlap too much with prior 2 methods.  (The shows similar to bakemonogatari are all bakemonogatari sequels)
In a production environment, multi-arm bandit testing would help tweak these hyperparameters (number of hidden factors, regularization).
The best results I have found so far are regularizing User factors but not anime factors.  These 'logical' groupings found have been with 56 hidden factors,  user l2(.0065), anime l2(0)
It's important to note that because there are two alternative methods which provide 'traditional' answers, this can be used to help with 'novel' or 'wildcard' recs. If this was the core of a recommende system without either of the other two item-item similarity methods, this would be better left with less hidden factors and less regularization.

Here are some examples of these 'wildcard' similarities
![lfgirlsundpanz.png](/readmepics/lfgirlsundpanz.png?raw=true)
![lfbake.png](/readmepics/lfbake.png?raw=true)
![lfdeathnote.png](/readmepics/lfdeathnote.png?raw=true)

# Predicting User Scores
### 3 Neurals Nets for collaborative filtering

There are 3 pre-trained neural nets. Each net has been trained to predict one type of score: Score, Usr Scaled Score, Anime Scaled score.
The Nets are loaded with their trained weights, only the user embedding layer is unlocked .
Each neural net predicts the user’s score for every possible rec (from the item-item methods). These predicted scores are sorted to give recommendations.

Every neural net has the same architecture:

Architecture:
![nn1_1.png](/readmepics/final_model.png?raw=true)
##### Neural Net 1: Predicting user score
Data Prep:All scores of 0 (status but no score) are dropped. Scores from outlier (<25 scores, >1500 scores) users are dropped.  
No Modifications are made for Watching, On-hold, Dropped, etc.
Both users and Animes are embedded with N hidden factors.  I use 36 factors in the final model.  (These factors are not shared between models)

The point of this neural net is simply try to chase the highest score (Completed, In Progress and Want to Watch will be ignored in this net). Because it is trained against the entire dataset, this will tilt towards 'popular' shows.

##### Neural Net 2: Only 'completed' or 'Dropped' shows matters: User_scaled Score
Data Prep:All scores not listed with a status of ‘Completed’ or ‘Dropped’ are ignored.  Scores from outlier (<25 scores, >1500 scores) users are dropped.
Both users and Animes are embedded with 36 latent factors.  (These factors are not shared between models)
On the 1 million score test set, the best MAE was: 0.916866
The final MAE was 0.949197
This neural net is trying to predict shows that users will like more than their personal average.

##### Neural Net 3: Only 'completed' or 'Dropped' shows matters: User_scaled Score
Data Prep:All scores not listed with a status of ‘Completed’ or ‘Dropped’ are ignored. Scores from outlier (<25 scores, >1500 scores) users are dropped.
Both users and Animes are embedded with 36 latent factors.  (These factors are not shared between models)
On the 1 million score test set, the best MAE was:0.932877
The final MAE was 0.98392

##### The Guilty Pleasure Variation: Predict completed rather than score
![nn1_1.png](/readmepics/anime_rec_arch_guilty.jpg?raw=true)

New Hyper-Parameter( Max # of reviews in the dataset to be recommended):
The Neural Net has the same architecture as the score predictor but with a sigmoid function on the final layer.
Trained on the entire 22 million User, Anime, Status data.
This model requires different hyperparameter tuning than the score predictors:
The best I've found so far:
n_highest_rated_shows= 7
k_nearest_shows = 5
t_highest_predicted_score=15
l_max_reviews = 30000
