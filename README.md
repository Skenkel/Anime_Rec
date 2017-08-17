
#  Anime_Rec
##  A Data Science Portfolio Project by Sam Kenkel

Purpose: Build a recommendation engine to recommend Anime (Japanese Cartoons) based on the existing data on MyAnimeList.net.

Anime fans (or 'Otaku', as they prefert to be called), like consumers of any media, can use assistance from modern machine learning and data science techniques in deciding Anime to watch.

Additionally, they have been kind enough to provide a large data set of public user scores.



## The Data
#### Collection Process
A number of scripts are used to gather data,  becaue MyAnimeList (referred to as mal from here on out) has a history of DDOS attacks, they no longer give out production API keys.
As such, all data for this project was acquired via scraping.

##### Usernames
MAL does not give any way to enumerate users, except via a search function.
MAL_username Scrape generates a pseudorandom collection of valid mal usernames by randomly choosing a three digit alphanumeric string, searching for usernames containing that string, getting every user, and repeating.
I stopped collecting usernames once I had 1.5 million user_names, as I deemed that to be a large enough sample to allow for futher data collection.

##### User Scores
Once I have a set of usernames (A list->set-> list conversion removes duplicates nicely) Mal-Score-Scraper is used.
This script import as uses Mal-Scraper (https://github.com/QasimK/mal-scraper ) to get scores (a number from 1-10) and status (Watching, Completed, On Hold, etc) for each user.  Once I have this information I use pandas to save it to a dataframe in memory, then write out to a csv.
It's important to note that I parralelized this aspect of the scrape (with three systems using a vpn and different vpn endpoints) and that this could be easily parrelelized to an arbitrary number of systems.



The raw data of user scores looks like:
![Screenshot%20from%202017-07-25%2009-01-00.png](Screenshot%20from%202017-07-25%2009-01-00.png?raw=true)

After running a scraper, I can convert this to a data frame:

![dataframe.png](dataframe.png?raw=true)

Anime_Score_Scraper.ipynb shows the code I used for this.
It's important to note that I have a UNKEY (Which is unique key) field. This allows for multiple score files to be merged later and duplicates easily removed.


```python

```

### Reviews
Users can give reviews of shows:
![review.png](review.png?raw=true)

We can scrape into a dataframe
![reviewframe.png](reviewframe.png?raw=true)

This gives me plaintex reviews of 6000 out of 11000 shows (the other 5000 shows lack reviews because they are too obscure for a user to have actively written a review, another example of nonresponse bias in this dataset).  
Anime_Review_scraper is used for this purpose. To prep for this, another tool (Anime_Info_Scraper) had to pull the URLs of every Anime's review page. (While each Anime's info page can be found by using myanimelist.net/ANIME_ID/
to get to an anime's review page  you must have the correct myanimelist.net/ANIME_ID/Anime_NAME/Reviews address.
Anime_Info_Scrape was used to generate that url, get a a description and genre for every anime, and get the url of the user-recommendations pages (for future use).


```python

```

# EDA
### Exploratory Data analysis
We can look at this data, and notice some important trends:

Here are scores with the 0's removed.
![MalScores.png](MalScores.png?raw=true)

Here is a comparison of average scores based on status.
![score%20by%20status.png](score%20by%20status.png?raw=true)

AVG Scores a user gives as a function of how many reviews the user has done.
![reviews%20per%20user.png](reviews%20per%20user.png?raw=true)

Notably:
There are signficant differences in how otaku score shows that they have not seen, but plan to, are seeing, have seen, and decided to stop watching.

In the Data, any time a user indicates they are watching, will watch (or have begun watching but stopped) a show without giving it a score is represented by a 0.



# Data (Continued)
### Other Notes:
It's important to note that all of this data is explicit user feedback from a site that requires users to register in order to rate shows.

This means that the "Average" user of this site is likely not a casual fan of Anime (The 'nonresponse' bias of fans who are not willing to register on a website is significant).  Any analysis, or modeling done on this data may not apply to casual Anime fans.

The AVG scores as a function of user reviews indicates we should be concerned that Mal may have  some of the same issues that have been seen with  imdb: people creating 'Dummy' accounts just to vote up specific shows.

Finally, as with all explicit feedback based data on media 'Guilty Pleasures' are not modelled well. Anime that is popular and controversial, but that Otaku do not rate highly in a public forum, will not be well served.

The data  does not have any time elements to it, so seperating scores that users have given to shows recently, or shows that they cannot have seen (because they have not yet aired) or shows that they likely have seen years ago is impossible with the data available.

# Architecture
As a brief explanation of the overall architecture,
Once the Score and Review scrapers have been used to generate full data-lake of user scores and Anime Reviews, there is multi-step process used to generate a recommendation.  To show the larger architecture before going into the components:

![anime_rec_arch3.jpg](anime_rec_arch3.jpg)

In short, Each user's 10 highest scored shows are used (substituting a static list of 'seed' shows if a user has scored less than 10 shows. Ties are broken by prioritizing the least popular shows (as defined by the MAL 'Number of users' field which was scraped with the anime info scraper.)

The static list I am using is Bakemonogatari, FMA:B, Steinsgate, AoT, Eva, Galactic Heroes, Haikyuu, Code Geass, To-Love Ru, Nisekoi, but this is entirely personal preference.  In a 'production' environment these would be determined by multi-arm bandit testing, or by business needs.

Each of these 10 seed shows is fed to three different item-item similarity filters, which return the 10 most similar anime (ignoring anime that the user has marked 'complete' and proceeding down the similarity list until 10 shows are generated).

This generates between 10 (if the 10 most similar non-completed anime from each of the three filters were the same, for every show) and 300 (if there is no overlap) of 'possible recommendations'.
In a production environment, new releases, or a static list of shows could be inserted as this step.

The 'possible recommendations' are all fed to 5 different user score prediction  systems (Neural Nets).
4 of these neural nets have the same architecture, but are trained with user-anime-score data that has been processed differently, and the 5th has the data presented in a way that it is modified.
Each of these networks predict's the user's score for each of  'possible recommendations', and returns a ranked list.
These ranked lists are ensembled (taking the top 2 from each neural net, detecting duplicates and proceeding down the ranked list), until there are 10 anime.
Those 10 anime are the recommendations for the user.

# Generating Possible Recommendations
### Three Types of Item-Item Similarity filtering

##### Type 1: User Score Cosine Distance collaborative filtering
Anime_Score_Sim shows the code for this.
First, we drop all 0's.
Next, we find the average score for each user, and subtract that score from every user's score. This is normalize the data (Without doing this, user's having different ideas of what constitutes a 6 vs a 7 vs 8, adds a lot of noise to the data).
Then make a sparse matrix of every user x every anime (with the user's scaled scores being at the intersection).
After some remapping to get those into anime names for eda, we can spot check to confirm that our similar anime are logical:
![users_cosim_deathnote.png](users_cosim_deathnote.png?raw=true)
If we look at Death Note, one of the most popular and succesful anime of all time, we get a number of other very popular and succesful shows, including Attack On Titan (which is directed by the same director, Tetsuro Araki). However, it's hard to judge these recommendations as this almost just a list of 'Crossover Anime'.

Let's look at something more obscure:
![user_score_cosim_girlsundpanzer.png](user_score_cosim_girlsundpanzer.png?raw=true)
Girls Und Panzer, the twee show about 'Cute Girls riding in Tanks as a highschool sport' is a little more informative. We see a great number of spin-off and sequel works (MAL lists prequels and sequels as different anime), as well as a number of other 'Cute Girls doing Cute Things shows', such as Hidamari Sketch  and Non Non Biyori.

If we examine the polarizing show Bakemonogatari:
![user_score_cosrim_bakemono.png](user_score_cosrim_bakemono.png?raw=true)
We see one of the downside of this form of recomendation system.  We see 10 recomendations for the sequels to Bakemonogatari, and 1 recomendation for another work by the same author (Katanagatari). While this certainly indicates that these are similar, shows, it is likely that using this system of similarity with the MAL dataset will give 'possible recs' that the user has allready seen, but not scored, or is already aware of.

##### Type 2: Anime Review  NLP TF-IDF   Cosine Distance collaborative filtering
Review_Sim shows the code for this.
We begin by merging all of our reviews  into a one dataframe and removin duplicates.
Then we generate a dataframe that consists of each row containing one  anime with every review's text merged together.
Then we use a TF-IDF (Term Frequency, Inverse Document Frequency) vectorizer generate a sparse matrix for each anime. We use some custom stop-words (such as anime), which we want the vectorizer to ignore.
N-grams are set from 1-3, which means that two word phrases (Such as 'moe trash', are treated as a word to vecotrized and counted)
In short, this is taking every 1 to 3 word phrase in every review, determining how often that 'word' appears in review text for one anime (such as cowboy bebop) and compares it to how often that 'word' appears in all reviews.

There are few ways we can vecotrize this, namely, we can do all words, with all counts, all words with a binary(count of 0 or 1) or we can slice this down to 900 most common words.
To clean up the text before we vectorize it, we remove some common stop words, and replace that name of the anime with the word 'anime_name' (which is also ignored)
With this sparse matrix of values, we can once again compute cosine pairwise distance between all shows. (So shows which are described by anime fans using similar uncommon words will be closer to each other).
We then turn these similarity scores into a dataframe which allows us to look at shows most similar to each other.

Then we check the results.
For Death Note we see:
![tfidf3deathnote.png](tfidf3deathnote.png?raw=true)
A who's who of very popular, well liked anime, but this is a different list than user score distance gave us. (Specifically, this list tilts towards older, 'darker' shows.

If we check Girls Und Panzer:
![tfidf3girlsundpanzer.png](tfidf3girlsundpanzer.png?raw=true)
we see excellent results. This list is not only different from the user score distances, but is a wide variety of "Moe" (Cute girls doing cute things) shows, with the highest scored show being "High School Fleet" which is a show that has the same premise, but with battleships rather than tanks.

When we look at Bakemonogatari,
![tfidf3bakemonogatari.png](tfidf3bakemonogatari.png?raw=true)
we also see a marked improvement. We see the shows direct sequel (Nisemonogatari), and a show based on a novel by the same author (Katanagatari), but we see a number of similar shows (Durara, Spice and Wolf, Mawaru Penguindrum) that are different from what user score distance returned.

##### Type 3: Aime Hidden Factors( HF)    Cosine Distance collaborative filtering
Mal_Keras_Fm_Clustering shows the code for this.
What I do is make a factorization machine neural net. This neural net embeds each user and each anime as N hidden factors, then makes the score a simple dot product of these hidden factors.
After training this neural network, we use the embedding for each show (which are the shows 'hidden factors') and do cosine pairwise distance.
By changing the number of hidden factors, and the amount of regularization, we can modify how 'hidden' these factors become.

Here is the network that is used to generate this

![dot2.png](dot2.png?raw=true)

The more Latent factors and the heavier the regularization, the stranger the similar anime become. Without regularization and too few hidden factors the results overlap too much with prior 2 methods.  (The shows similar to bakemonogatari and are all bakemonogatari sequels)
In a production environment, multi-arm bandit testing would help tweak these hyperpameters (number of hidden factors, regularization).
The best results I have found so far are regularizing User factors but not anime factors.  These 'logical' groupings found have been with 36 hidden factors,  user l2(.005), anime l2(0)
It's important to note that because there are two alternative methods which provide 'traditional' answers, this can be used to help with 'novel' or 'wildcard' recs. If this was the core of a recomender system without either of the two item-item simimlarity methods, this would be better left with less hidden factors and less regularization.

Here are some examples of these 'wildcard' similarities
![lfgirlsundpanz.png](lfgirlsundpanz.png?raw=true)
![lfbake.png](lfbake.png?raw=true)
![lfdeathnote.png](lfdeathnote.png?raw=true)

# Predicting User Scores
### 5 Neurals Nets for collaborative filtering

There are 5 different neural nets, which are all trying to predict user scores.
4 of these neural nets have the same architecure, but because of different data processing steps, the neural net will make different predictions.
##### Neural Net 1: Only the score matters
Data processing:All scores of 0 are dropped.
No Modifications are made for Watching, On-hold, Dropped, etc.
Both users and Animes are embedded with N(Currently set to 40) hidden factors.

Architecture:
![nn1_1%20%281%29.png](nn1_1%20%281%29.png?raw=true)

The point of this neural net is simply try to chase the highest score (Completed, In Progress and Want to Watch will be ignored in this net).

##### Neural Net 2: Only 'completed' shows matters
Data processing:All scores of shows not 'completed' are dropped.

This uses the same architecure as NN1.

The intention of this Neural Net is to get 'cleaner' data by ignoring all of the noise that comes from any show the user hasn't completed.


##### Neural Net 3: Heavy modification of scores

All nonzero scores are modified:'COMPLETED':0,'CONSUMING':0,'DROPPED':-2,'ONHOLD':-1,'BACKLOG':-.5

The purpose of this data prep is train a network that predicts Anime a user is likely to complete. This is because a user who drops a show and does not give it a score is treated as having given that show a -2 on a 10 point scale.

##### Neural Net 4: Score Imputation
All zeros scores are imputed:'COMPLETED':8,'CONSUMING':7,'DROPPED':3,'ONHOLD':5,'BACKLOG':6
All nonzero scores are unmodified.
This has the same purpose as network 3 (train the network that incompleted shows are worse than shows that users have completed), but it does so without the staggering cost of negative scores being introduced.


##### Neural Net 5: Status Embedding
All scores are left alone.
The Neural Net is fed with user, score, and status(all of which are embedded to hidden factors). As a result, the neural net learns the interactions between users, animes and status (users may drop Gintama while still liking it).
This is the architecure of this network:
![nn5_1.png](nn5_1.png?raw=true)
