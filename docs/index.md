It's no secret that Australians love their sport. If you go to any social gathering here, it's highly likely that you'll catch at least a snippet of a conversation about sport, whether it be cricket, tennis, soccer or some other sport. However nothing quite matches Australians' passion for Australian Rules Football. The AFL (Australian Football League) is the sport's highest-level league and regularly attracts match crowds in excess of 60,000 people.

![Image](/Mark.jpg)
-> A Collingwood player takes a spectacular mark in the 2018 Preliminary Final <-


Whilst watching a blockbuster match one day, I noticed that the attendance was significantly less than expected for such a highly-anticipated game. This got me thinking about all the possible factors that influence crowd sizes, from the ladder positions of the teams to the match-day weather conditions. Being the kind of guy who loves data-backed answers to questions, I decided to make a machine learning model for it using the Python programming language.

An AFL match attendance predictor could be valuable for companies and key decision makers in a variety of industries. Some use cases include:

- Forecasting stadium revenue
- Valuing stadium advertising opportunities
- Logistics and staff planning at stadiums
- Forecasting and planning for independent businesses (e.g. restaurants and cafés) who operate near stadiums

In fact, I came across an internet forum post from somebody who was operating a café near Marvel Stadium in Melbourne and was searching for a tool to estimate the amount of foot-traffic his café would receive on future AFL match days (a suitable proxy for this would be the attendance at Marvel Stadium games). Here's what he wrote:

_"Could anyone point me in the right direction of where I would find an estimated crowd attendance figure in advance before the game, particularly for [Marvel] Stadium? My reason being, I currently manage a cafe near the stadium and would like to get an idea of the possible foot traffic passing my cafe on game day."_

In this blog post, I present a non-technical discussion of my analysis and the models that I created, along with a summary of the results.

> _For readers who aren't familiar with the typical machine learning workflow, I'll use an analogy of a school teacher teaching a student a new concept to help you understand the core idea behind each step of the process._


![Image](/MCG crowd.jpg)
-> Almost 100,000 fans packed into the MCG for the 2012 AFL Grand Final <-

* * *

### Gathering the data

>_In the teacher-student analogy, the data gathering phase is like the teacher finding some notes, tutorials and exercises to give his student in order to help him learn the new concept._

Since I came up with the idea of this project myself, I had to find and collect the necessary data from various online sources. This process involved making a list of all variables that I expected to influence attendances and then collecting the required data. I ended up using 7 different data sources (listed in the data sources section at the end of this article) to obtain the following data for each AFL match between 2004–2018 (15 seasons in total):

- Start-time, date and round
- Stadium name and its capacity
- Teams' membership numbers and ladder positions
- Daily rainfall for the relevant city/town
- Australia's population for the corresponding year

* * *

### Data cleaning and preparation

> _Going back to the analogy, the data cleaning/preparation stage is like the teacher correcting any errors in his notes, removing any irrelevant exercises and organising all the study materials into a neat textbook for his student._

Once the raw data had been collected, I began the process of cleaning and collating it into a single dataset. Some of the noteworthy steps of this process were:

**1. Converting between time zones**

One of the data files that contained match start-times listed all times in AEST (the time zone of Australia's East coast), even for matches that were played in one of Australia's two other times zones. To fix this, I wrote a simple function to convert all the time values to local time.

**2. Changing the stadium and team names to match across data files**

AFL stadium names change regularly due to sponsorship agreements, so I changed all the names to a standard set to avoid issues later on. Also, some of the files had used nicknames for teams, so I altered these values to make them match across all files.

**3. Filling in missing rainfall values**

The rain data had 114 missing values for match-days, so I decided to impute these values by estimation. I designed a 2-step process to do this:
- **Step 1:** Try to estimate the rainfall value by taking an average of the surrounding 4 days' rainfall (i.e the 2 preceding days and 2 following days). If there are less than 2 values in this 4-day window, then this method wouldn't give an accurate estimate (and won't give any estimate at all if there aren't any values in this window), so when this was the case, step 2 was invoked.
- **Step 2:** Estimate the missing value from the corresponding city's rainfall data by using the average rainfall for that particular month (e.g. Sydney's average daily rainfall for June).

**4. Converting the _attendance_ variable into _percentage-filled_**

Instead of aiming to predict match attendance, I focused on the percentage-filled metric (i.e. what percentage of available seats were filled) for two main reasons. Firstly, due to the large variability in AFL stadium capacities, using percentage-filled would lead to more stable predictions and would help with interpreting the model's key performance metrics. Secondly, some of the stadium capacities varied over time due to renovations and this obviously posed a problem to predicting the actual attendance figure.

**5. Combining the 7 data files into a single dataset**
I started with the fundamental dataset (match details and attendance) and successively added in new variables from the other data files (e.g. rainfall, team memberships and stadium capacities). These operations were made particularly efficient by taking advantage of multi-indexing, which enabled me to use Pandas's optimised _apply_ function.

* * * 

### Feature engineering

>_The feature engineering stage is where the teacher alters some of the exercises in his textbook to tailor them specifically to the student's learning style, with the aim of making the teaching/learning process more effective._

Below I explain the main feature engineering steps that I carried out.

**1. Converting _round_ into a percentage value**

The number of rounds in an AFL season has varied over the past 15 years. To account for these inconsistencies, I decided to convert the _round_ variable to _round percentage_. For example, if a Round 11 match took place in a season that included 22 rounds in total, then the match's _round percentage_ value would be 50%.

**2. Creating a _time-slot_ variable**

AFL matches are played in relatively well-defined time-slots (e.g. Friday night, Saturday afternoon and Sunday twilight), and most fans would agree that the time-slot influences the attendance. Many machine learning algorithms wouldn't be able to pick up this nuance simply by taking in raw date and time information, so I decided to create a dedicated variable for each match's time-slot.

**3. Adding indicator variables for finals and other special matches**

It's obvious that finals matches (sometimes referred to as _playoffs_ in other sports) attract significantly larger crowds that normal games. To introduce this important information into my dataset, I created separate variables to indicate whether or not a match was a particular type of final (grand final, preliminary final, etc). I also created indicator variables for "special" games that usually attract larger-than-normal crowds, e.g. ANZAC Day games and Western Derbies.

**4. Adding combinations of features**

I knew that certain models would benefit from being hand-fed informative features such as _team closeness_ (games between closely-matched teams are expected to draw larger crowds), _sum of positions_ (matches that feature two high-quality teams are expected to draw larger crowds) and _total members_ (this provides a way to quantify the total supporter-base, instead of just looking at the home and away teams' supporters in isolation).

**5. Putting rainfall values into categorical "bins"**

I decided that it would be valuable to categorise the rainfall values into categories because:
  - There were some very large values that would act as high leverage (very influential) points and cause issues in some regression models
  - Categorisation improves model usability (users may not know the exact expected rainfall for a given day, but could estimate whether any rain is "light", "moderate", etc)
  - It also improves model interpretability

The cut-offs for the categories were chosen using the 0.33 and 0.66 quantiles:
  1. No (or barely noticeable) rain: 0–2mm
  2. Light rain: 2 – 3.6mm
  3. Moderate rain: 3.6 – 7.6 mm
  4. Heavy rain: > 7.6mm

**6. One-hot encoding categorical variables**

Most machine learning algorithms can't deal with categorical variables (stadium name, time-slot, etc) directly, so I used a technique called one-hot encoding to transform them into numerical representations.


After completing the feature engineering stage, I created some data visualisations to gain an understanding of how the predictor variables were associated with the response variable. See below for some of these plots.

![Image](/Stadiums.png)
->Average attendance percentage for each stadium<-


![Image](/Team Closeness.png)
->Team closeness (i.e. difference in teams' ladder positions) plotted against average attendance percentage<-


![Image](/Timeslot Attendances.png)
->Average attendance percentage for each time-slot<-


* * * 

### Model building

>_The model building phase is like the teacher using his newly-created textbook to teach his student. He tries out a variety of different teaching techniques (e.g. whiteboard explanations, videos and quizzes) to see which technique works best in teaching his student the new concept._

The final step was to train a few different machine learning algorithms using my prepared data and evaluate each of them to choose the "best" one. I chose to use 5-fold cross validation with the metric of RMSE (root mean squared error) to assess each model's accuracy.

**1. Linear Regression**

Linear regression is a simple algorithm that usually produces models that are more interpretable but less accurate than more advanced algorithms. The resulting coefficients of my linear regression model are shown below (note that the baseline factors for the _time-slot_ and _stadium_ dummy variables were Saturday night and the MCG, respectively).

![Image](/Linear Regression Coefficients.jpg)
->Feature coefficients from the linear regression model<-


The coefficient for a variable _x_ represents the estimated change in _percentage attendance_ that would occur if _x_ were to be increased by one unit (with all other variables held constant). For dummy variables, the coefficient represents the change that would occur if the match met that specific condition (e.g. played at Adelaide Oval) as opposed to the baseline factor.

The linear regression model achieved a cross-validated RMSE of 12.96. Below is a graph that shows the predictions plotted against the actual values.


![Image](/actual vs predicted.png)
-> The model's predictions plotted against the actual values <-


**2. LASSO Regression**

LASSO is a regularisation method that attempts to build sparse models from a linear regression by shrinking the variables' coefficients (it usually makes the model more sparse by setting the least important variables' coefficients to zero). This is helpful to avoid _over-fitting_ and also to enhance interpretability.
I first standardised my features so that they each had mean 0 and standard deviation 1, and chose the regularisation parameter through 5-fold cross validation. The final LASSO model had chosen to ignore 3 variables: _away members_, _away ladder position_ and the _Saturday twilight_ dummy variable. Its cross-validated RMSE was 12.66 (a slight improvement over the linear regression model).

**3. Regression Tree and Random Forest**

Decision trees are models that work similarly to how humans would make predictions in real life. They ask a series of yes/no questions and follow a flow-chart based on the answers, eventually arriving at a prediction. Below is a plot of a basic decision tree that I built using my AFL data. As an example of how it works, let's say I wanted to use this tree to predict the attendance of the Round 20 game between West Coast and Fremantle in 2015. I'd end up predicting an attendance of 95.2% (i.e. 41,014 people) because:
  - The sum of the teams' ladder positions was less than 18
  - The round completion percentage was less than 97%
  - The game was a Western Derby

If you're curious to know, the _actual attendance_ of that game was 41,959.

![Image](/Regression Tree.png)
->A simple decision tree for predicting percentage attendance at AFL games<-


This tree could have been made more complex by adding more layers (and hence allowing for greater flexibility in predictions) however this leads to over-fitting. To overcome this limitation, I tried out a powerful algorithm called _random forest_ that creates many of these trees and takes the average of their individual predictions (there are a few tricks involved that make it such a powerful algorithm, but I won't go into those details here).

I tuned 5 different hyperparameters of the random forest model using a technique called _randomised grid search_ and trained the final model using the best combination of parameters found. The model achieved a cross-validated RMSE of 13.47 (this was quite surprising, as I expected it to perform better than the LASSO regression model).

**4. k-Nearest Neighbours Regression with PCA**

k-Nearest Neighbours is an algorithm that makes a prediction for a specific observation by taking the average of the _k_ most "similar" samples in the dataset. It's often a good approach when the number of variables is small, but since I had a relatively large number of variables, I chose to incorporate a technique called _PCA (Principal Component Analysis)._

Building the k-Nearest Neighbours model with PCA involved 4 key steps:
  1. Standardising the features to have mean 0 and variance 1
  2. Extracting the principal components
  3. Selecting the number of principal components and the number of neighbours to use in the model via cross-validation
  4. Training the algorithm using the two optimal values found in step 3

The final model achieved a cross-validated RMSE of 13.03.

* * *

## Summary

This project involved 4 important stages: data collection, data cleaning/preparation, feature engineering and model building. The best-performing model turned out to be the LASSO regression, but most of the other models achieved a similar level of accuracy. Future work could look into using more features such as social media sentiment on the teams and match-day temperature. A few different machine learning ensemble techniques (e.g. boosting, bagging and stacking) could also be used in an attempt to improve the prediction accuracy.

