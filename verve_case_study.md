# Verve Group data science case study

## Instructions

Thank you for undertaking our data science case study. The purpose of this case study is to help us to better understand your data science knowledge and your communication skills. This challenge is not timed, so you can take it whenever you have about an hour free in your day. Our expectation is for you to not spend more than one hour on this. Please send us your solution back within a week of receiving this case study. We ask that you please submit your answer in a public Github repository.

Below you will be given some information about a dataset. In order to contain the amount of time this task will take to complete, we’ve provided you with summary statistics and visualisations about this dataset rather than the raw data itself. The conclusions you draw about the data should be based on these summary metrics.

We ask that you complete the following tasks:

1. Imagine that you were asked to use this dataset to build a classification model, with `gender` as the target. Look at the information we have given you and identify 3-5 potential problems you can see with the provided dataset that might make building a classification model difficult.
2. Describe briefly how you would find the features that are likely to be the most important for your model.
3. Identify which model you would try first, and at least one advantage and disadvantage of this choice.
4. Write up your findings in a Markdown document.
5. Create a new Github repo and commit your Markdown document there.

Thank you, and good luck!

## Overview of the data

In order to monetise, publishers of apps sell ad space to advertisers. For example, when scrolling through your Instagram feed, you might see an ad like the one below. 

![Flow of problem](https://user-images.githubusercontent.com/5158813/123070031-b325d400-d413-11eb-9139-e674d164b6b3.jpeg)

Every time a user interacts with an app and is shown an ad, data are generated. An example of the sort of data you might get is shown in the diagram below. In this example, a user opens a fashion app and is shown an ad for clothing, which they click on. We also get information about how long the user spent in the app in total, as well as the name of their device. Each of these interactions are called events. These events are sent off the device and stored (e.g., in an S3 bucket).

![Flow of problem](https://user-images.githubusercontent.com/5158813/123070587-2f201c00-d414-11eb-8feb-11bf1b9ade6c.png)

Advertisers want to show ads to the most relevant users, in order to increase their chances of getting a click. As such, they are very interested in targeting only the most relevant users based on characteristics such as their gender, age, income level and interests.

In the data science team, we've been tasked with trying to **predict the gender (male/female)** of a device user in order to help advertisers to target within apps.

### Features

* `device_name`: the name given to the device, e.g., "Maria's iPhone"
* `app_category`: the classification of the app content, e.g., "fashion"
* `interaction_with_app`: how long the user interacted with the app during this session (in minutes)
* `ad_category`: the classification of the ad that was shown during the event (e.g., "clothing")
* `click`: whether the user clicked the ad (yes/no)

### Target

* `gender`: the gender of the device user (male/female)

### The unit of analysis

Each row of data represents an **event**, where a user is shown an ad during an app. Users can have multiple events for the same app session, as each time they are shown an ad while using the app it is logged as a new event. Moreover, a user can have events from different apps.

| user_id | app_id | device_name      | app_category        | interaction_with_app | ad_category           | click | gender |
| ------- | ------ | ---------------- | ------------------- | -------------------- | --------------------- | ----- | ------ |
| 10043   | 26784  | Anna's phone     | Health              | 8                    | Prescription medicine | No    | F      |
| 10043   | 45278  | Anna's phone     | Sports              | 12                   | Luxury cars           | No    | F      |
| 37489   | 25361  |                  | Desserts and baking | 21                   | Restaurants           | Yes   | F      |
| 20748   | 83647  | iPhone de Manuel | Automotive          | 18                   | Mid-range cars        | No    | M      |
| 73947   | 83647  |                  | Automotive          | 2                    | Luxury cars           | No    | M      |

### Sample size

The dataset has 3700 rows.

## Summary statistics

### Outcome variable

```
Frequencies for variable gender
          frequency  proportion
male          2663    0.719730
female        1037    0.280270
```

### Features

#### Device name

```
Frequencies for variable device_name
                  frequency  proportion
Has value             1513   0.408919
Value missing         2187   0.591081

Examples of the output of device_name
Marco's iPhone
iPhone de Jose
Jane's phone
Vinitha's tablet
```

#### App category

```
Frequencies for variable app_category
                   frequency  proportion
News                  2015    0.544595
Weather                970    0.262162
Health                 339    0.091622
Dating                 154    0.041622
NaN                     66    0.017838
Automotive              49    0.013243
Sports                  37    0.010000
Fashion                 33    0.008919
Arts and crafts         20    0.005405
Desserts and baking     17    0.004595
```

#### Ad category

```
Frequencies for variable ad_category
                           frequency  proportion
Non-alcoholic beverages        787    0.212703
Clothing                       612    0.165405
Beauty                         363    0.098108
Mid-range cars                 343    0.092703
Beer and wine                  340    0.091892
Telecommunications             210    0.056757
Tax planning                   191    0.051622
Insurance                      179    0.048378
Jewelry                        170    0.045946
Credit/debt and loans          137    0.037027
Air travel                     136    0.036757
Restaurants                     94    0.025405
NaN                             66    0.017838
Prescription medicine           39    0.010541
Luxury cars                     33    0.008919
```

#### Click

```
Frequencies for variable click
         frequency  proportion
Yes           18    0.004865  
No          3682    0.995135
```

#### Interaction with app

<img src="https://user-images.githubusercontent.com/5158813/123070690-4232ec00-d414-11eb-9e69-c74bde54b079.png" alt="Flow of problem" style="zoom:60%;" />

# Answers : 

## 1 - Potential problems :

#### The unbalance between classes in the target variable : 

- The `gender` variable has an unbalance in its classes (~72% male to ~28% female).
- We could end up with a severe unbalance in the above classes when grouping by the `user_id`.
- Undersampling or oversampling techniques can help build up a balanced dataset.

#### The missing values in the features :

- Mainly in the `device_name` feature, in which ~60% of the values are missing.
- Also in the `app_category` and the `ad_category`, but the missing values proportion is considered low, so they can be considered as another category or they can be imputed with the most probable category (with respect to the other features for example).

#### The unbalance between classes in the features : 

- The most noticeable unbalance is in the `Click` feature (~0.4% Yes to ~99.5% No).
- `app_category` and `ad_category` also have high frequency classes and low frequency classes.

#### Low number of rows after aggregation : 

- After grouping by the `user_id` feature, data may end up having a small number of rows.

## 2 - Interesting features : 

### Feature engineering : 

#### Extracting the name of the person from the device name and checking whether the name belongs to a male or a female : 

- Using the nltk library for example, we can extract the name of the person from the `device_name` feature.
- Using either a library (gender-guesser for example) or a classification model trained on names and genders, we can create another feature where we put the guess of the gender (based on the name) or unknown (if the `device_name` is missing).

#### Mean, min, max, std encoding (with respect to the gender) of the interaction with app :

- Patterns can be found within these features allowing to help with the classification.

#### Discretizing the interaction with app : 

- Creating an **ordinal** feature containing bins created from the discretization of the `interaction_with_app` feature.
- Bins can be for example : Short - Medium - Long ( or 0 - 1 - 2 )

#### Merging Ad categories that fall in a same "bigger" category : 

- Depending on the data (gender proportion for each category), we can merge categories in order to reduce the number of classes in the `ad_category` feature and increase the number of samples per class. 
- For example : merging Mid-range cars with luxury cars / merging Jewelry with Beauty.

#### Merging App categories that fall in a same "bigger" category : 

- Same reason and logic with previous statement.

#### Creating a feature that contain the information on both the app category and the ad category :

- Concatenating the `app_category` and the `ad_category` into a single feature.
- This may or may not be useful as it can create multiple classes.

#### Extracting the type of device from the device name : 

- The feature can for example contain : iPhone, phone, tablet, unknown.
- This feature may or may not be interesting since we have an important number of missing device names.

#### Extracting the language from the device name : 

- This feature can contain for example : French, English, unkwown.
- Knowing the language can give a hint on the country/culture and this can help with the prediction (gender tendencies may differ from a country/culture to another).
- Same as previous statement, this may or may not be interesting due to the number of missing device names.

#### Counting the number of apps a user uses : 

- Using the `user_id` and `app_id`, we can count the number of apps a user uses.
- This feature can also be discretized later on.
- Following the same logic, we can also extract the most used `app_category` by user.

#### Aggregating the whole dataset to make sure we have a unique user per row :

- When aggregating, categorical and ordinal features values can either be just copied (because they were the same for all rows for the same user) or picked based on the most frequent class (or they can be listed, for example all the `app_categories` that the user used, but this would create many classes within the feature, or many features in the case of one-hot encoding).
- Continuous features may also be copied for the same previous reason, or averaged.

#### Clustering feature : 

- Using an unsupervised learning algorithm (k-means with k=2 for example), we can add a cluster feature that can help with the gender classification.

### Feature selection : 

#### Removing highly correlated features

#### Using feature selection algorithms : 

- Using RFE (Recursive Feature Elimination) can help in keeping important features.

#### Using tree-based models : 

- Tree-based models can give an idea on interesting features because they internally calculate the feature importance. 

## 3 - First model to try : 

I would definitely go for a tree-based model as a first model and try out the **RandomForest** which will probably serve as a good baseline.

**Advantages :**

- It works very well on continuous and categorical variables.
- Can handle missing values in data.

**Disadvantages :**

- Performance of RF may not be great on imbalanced classification problems, because of bootstrap sampling (but this can be dealt with using the `sample_weight` or the `class_weight` parameters for example).
- Not a good choice if we are looking for interpretability.
