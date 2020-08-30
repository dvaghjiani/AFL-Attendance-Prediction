It's no secret that Australians love their sport. If you go to any social gathering here, it's highly likely that you'll catch at least a snippet of a conversation about sport, whether it be cricket, tennis, soccer or some other sport. However nothing quite matches Australians' passion for Australian Rules Football. The AFL (Australian Football League) is the sport's highest-level league and regularly attracts match crowds in excess of 60,000 people.

![Image](/Mark.jpg)
->*A Collingwood player takes a spectacular mark in the 2018 Preliminary Final*<-


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
->*Almost 100,000 fans packed into the MCG for the 2012 AFL Grand Final*<-

* * *

### Gathering the data

> _In the teacher-student analogy, the data gathering phase is like the teacher finding some notes, tutorials and exercises to give his student in order to help him learn the new concept._

Since I came up with the idea of this project myself, I had to find and collect the necessary data from various online sources. This process involved making a list of all variables that I expected to influence attendances and then collecting the required data. I ended up using 7 different data sources (listed in the data sources section at the end of this article) to obtain the following data for each AFL match between 2004–2018 (15 seasons in total):

- Start-time, date and round
- Stadium name and its capacity
- Teams' membership numbers and ladder positions
- Daily rainfall for the relevant city/town
- Australia's population for the corresponding year

### Data cleaning and preparation

> _Going back to the analogy, the data cleaning/preparation stage is like the teacher correcting any errors in his notes, removing any irrelevant exercises and organising all the study materials into a neat textbook for his student._

Once the raw data had been collected, I began the process of cleaning and collating it into a single dataset. Some of the noteworthy steps of this process were:

**1. Converting between time zones:**

One of the data files that contained match start-times listed all times in AEST (the time zone of Australia's East coast), even for matches that were played in one of Australia's two other times zones. To fix this, I wrote a simple function to convert all the time values to local time.

**2. Changing the stadium and team names to match across data files:**

AFL stadium names change regularly due to sponsorship agreements, so I changed all the names to a standard set to avoid issues later on. Also, some of the files had used nicknames for teams, so I altered these values to make them match across all files.

**3. Filling in missing rainfall values:**

The rain data had 114 missing values for match-days, so I decided to impute these values by estimation. I designed a 2-step process to do this:
  - **Step 1:** Try to estimate the rainfall value by taking an average of the surrounding 4 days' rainfall (i.e the 2 preceding days and 2 following days). If there are less than 2 values in this 4-day window, then this method wouldn't give an accurate estimate (and won't give any estimate at all if there aren't any values in this window), so when this was the case, step 2 was invoked.
  - **Step 2:** Estimate the missing value from the corresponding city's rainfall data by using the average rainfall for that particular month (e.g. Sydney's average daily rainfall for June).

**4. Converting the attendance variable into percentage-filled:**

Instead of aiming to predict match attendance, I focused on the percentage-filled metric (i.e. what percentage of available seats were filled) for two main reasons. Firstly, due to the large variability in AFL stadium capacities, using percentage-filled would lead to more stable predictions and would help with interpreting the model's key performance metrics. Secondly, some of the stadium capacities varied over time due to renovations and this obviously posed a problem to predicting the actual attendance figure.

**5. Combining the 7 data files into a single dataset:**
I started with the fundamental dataset (match details and attendance) and successively added in new variables from the other data files (e.g. rainfall, team memberships and stadium capacities). These operations were made particularly efficient by taking advantage of multi-indexing, which enabled me to use Pandas's optimised apply function.

* * * 
### Feature engineering

>_The feature engineering stage is where the teacher alters some of the exercises in his textbook to tailor them specifically to the student's learning style, with the aim of making the teaching/learning process more effective._

Below I explain the main feature engineering steps that I carried out.

**1. Converting round into a percentage value**

The number of rounds in an AFL season has varied over the past 15 years. To account for these inconsistencies, I decided to convert the round variable to round percentage. For example, if a Round 11 match took place in a season that included 22 rounds in total, then the match's round percentage value would be 50%.

**2. Creating a time-slot variable**

AFL matches are played in relatively well-defined time-slots (e.g. Friday night, Saturday afternoon and Sunday twilight), and most fans would agree that the time-slot influences the attendance. Many machine learning algorithms wouldn't be able to pick up this nuance simply by taking in raw date and time information, so I decided to create a dedicated variable for each match's time-slot.

**3. Adding indicator variables for finals and other special matches**

It's obvious that finals matches (sometimes referred to as playoffs in other sports) attract significantly larger crowds that normal games. To introduce this important information into my dataset, I created separate variables to indicate whether or not a match was a particular type of final (grand final, preliminary final, etc). I also created indicator variables for "special" games that usually attract larger-than-normal crowds, e.g. ANZAC Day games and Western Derbies.

**4. Adding in combinations of features**

I knew that certain models would benefit from being hand-fed informative features such as team closeness (games between closely-matched teams are expected to draw larger crowds), sum of positions (matches that feature two high-quality teams are expected to draw larger crowds) and total members (this provides a way to quantify the total supporter-base, instead of just looking at the home and away teams' supporters in isolation).

**5. Putting rainfall values into categorical "bins"**

I decided that it would be valuable to categorise the rainfall values into categories because:
  - There were some very large values that would act as high leverage (very influential) points and cause issues in some regression models
  - Categorisation improves model usability (users may not know the exact expected rainfall for a given day, but could estimate whether any rain is "light", "moderate", etc)
  - It also improves model interpretability

The cut-offs for the categories were chosen using the 0.33 and 0.66 quantiles:
1. No (or barely noticeable) rain: 0–2mm
2. Light rain: 2–3.6mm
3. Moderate rain: 3.6–7.6 mm
4. Heavy rain: > 7.6mm

**6. One-hot encoding categorical variables**

Most machine learning algorithms can't deal with categorical variables (stadium name, time-slot, etc) directly, so I used a technique called one-hot encoding to transform them into numerical representations.


## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/dvaghjiani/Loan-Default-Prediction-Project/edit/master/docs/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/dvaghjiani/Loan-Default-Prediction-Project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
