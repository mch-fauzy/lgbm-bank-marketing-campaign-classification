# **Bank Marketing Campaign Classification**
| **Report** | **Script** |
|:---:|:---:|
| **[PDF](https://drive.google.com/file/d/15lNNkDXFw3ePQ5ge7EvQFP93LvOxlXpo/view?usp=drive_link)** | **[Jupyter Notebook](https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/Classification_Bank%20Marketing%20Campaign.ipynb)** |


# Table of Contents

[Business Understanding](#Business-Understanding)<br>
[Data Cleaning and Preparation](#Data-Cleaning-and-Preparation)<br>
[Data Preprocessing](#Data-Preprocessing)<br>
[Model Building & Evaluation](#Model-Building-and-Evaluation)<br>
[Model Explainability](#Model-Explainability)<br>
[Conclusion and Recommendation](#Conclusion-and-Recommendation)<br>

# Business Understanding
## Context
Based on [Investopedia](https://www.investopedia.com/terms/t/termdeposit.asp), a `term deposit` is a `fixed-term investment` that includes the deposit of money into an account at a financial institution. Term deposit investments usually carry `short-term maturities` ranging from `one month to a few years` and will have varying levels of required minimum deposits.

When an account holder `deposits` funds at a bank, the bank can `use that money` to `lend` to other consumers or businesses. `In return` for the right to use these funds for lending, they will `pay the depositor compensation` in the form of `interest` on the account balance. 

> A customer will `deposit or invest` in one of these accounts, agreeing `not to withdraw` their funds for a `fixed period in return` for a `higher rate of interest` paid on the account.

> The `interest earned` on a term deposit account is `slightly higher` than that paid on standard savings or interest-bearing checking accounts. The increased rate is `because access to the money is limited` for the `timeframe` of the term deposit.

> Term deposits are an `extremely safe investment` and are therefore very appealing to conservative, low-risk investors

For example
* A lender (bank) may offer a 2% rate for term deposits with a two-year maturity. 
* The `funds deposited` are then `structured as loans to borrowers` who are `charged 7% in interest` on those notes. 
* This difference in rates means that the `bank makes a net 5% return`. 
* The spread between the rate the bank pays its customers for deposits and the rate it charges its borrowers is called `net interest margin`. Net interest margin is a profitability metric for banks.

Banks are businesses, as such, they want to pay the `lowest rate possible for term deposits` and `charge a much higher rate to borrowers for loans`. This practice increases their margins or profitability. 
> However, there is a `balance the bank needs to maintain`. If it `pays too little interest`, it `won't attract new customers` into the term deposit accounts. Also, if they `charge too high` of a rate on loans, it `won't attract new borrowers`.

In `periods of rising interest rates`, consumers are more `likely to purchase term deposits` since the `increased cost of borrowing` makes `savings (deposit) more attractive`. Also, with higher market interest rates, the financial institution will need to offer the customers a higher rate of interest, so the customers also earns more.

When `interest rates decrease`, consumers are `encouraged to borrow and spend more`, thereby stimulating the economy. In a `low interest rate` environment, `demand for term deposits` can `decrease` since customers can typically find alternative investment vehicles that pay a higher rate.

> `Interest rates` should be `proportional` to the `time until maturity`, and the `minimum amount of principal lent` to the credit union or bank. 

> In other words, a `six-month term deposit` will likely pay a `lower interest rate` than a `two-year term deposit`. Customer `not only` receive a `higher rate` for `locking up their money` with the bank for extended periods, but also should `earn a higher rate for large deposits`. 

Based on the theoretical research [Factors Affecting Savings Deposit Decision of Individual Customers](https://www.researchgate.net/publication/342864549_Factors_Affecting_Savings_Deposit_Decision_of_Individual_Customers_Empirical_Evidence_from_Vietnamese_Commercial_Banks)  by Bui Nhat Vuong and team, the authors have looked at the model of the theory of consumer behavior.

<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/deposit_customer_behavior_model.jpeg">

Nevertheless, as business entities with financial products and respective customers, banks still have to compete to not lose customers. One of the ways to `acquire new customers` is by `conducting a marketing campaign`, in this dataset based on `customers demographic` and `Employee Knowledge`.


## Problem Statement
`Marketing` plays an increasingly important role, it has become one of the `factors` that directly `influence` the customer’s deposit decision. Promotions `contribute` to the customer’s decisions because if the bank puts out impressive promotions, creating benefits for customers, it will `attract` customers attention.

The bank wants to analyze the `effectiveness` of its marketing campaign for a term deposit subscription to determine whether the campaign was successful or not.

## Goals
To identify the `factors` that `influence` the customer's decision and to build a `classification model` that can `reduce the cost` of marketing.
* Understand the `characteristics of customers` who subscribed to the term deposit and those who did not.
* Determine which marketing campaign `channels` and `season` were the most effective in attracting customers.
* Develop `insights and recommendations` for improving future marketing campaigns.

## Analytics Approach
The analytics approach could involve using `descriptive statistics and data visualization` techniques to explore the data and `identify trends and patterns`. We could also use machine learning techniques such as classification to build `predictive models` that can help us understand which `factors` are most `important` in `predicting` whether a customer will make a `deposit`.
* Data Cleaning: Check for missing values, duplicate rows, and anomalies in the data.
* Data Preprocessing: Check for outlier, create new features from existing variables that may help in modelling, remove redundant feature, and apply feature selection.
* Model Building: Build predictive models to identify the factors that influence the term deposit subscription and to predict the likelihood of a customer subscribing to the term deposit.
* Model Evaluation: Evaluate the performance of the models using appropriate evaluation metrics.
* Insights and Recommendations: Based on the analysis, develop insights and recommendations for improving future marketing campaigns.

## Evaluation Metrics
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/confusion_matrix.png">

* Recall (Sensitivity): The proportion of `true positive predictions` out of all `actual positive` cases. In other words, recall measures how good the classifier is at `finding all` the `positive instances (subscribe)`.
   * The `lower the recall`, the `higher` the `loss` of `potential customers`.
    
* Specificity: The proportion of `true negative predictions` out of all `actual negative` cases. In other words, specificity measures how good the classifier is at `finding all` the `negative instances (not subscribe)`.
   * The `higher the specificity`, the `lower the marketing costs`, especially for `non-potential customers`.
    
* ROC AUC Score: The area under the receiver operating characteristic curve, which measures the ability of the `model` to `distinguish` between positive (subscribe) and negative (not subscribe) cases with ROC-AUC above 70%

# Data Cleaning and Preparation
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/data_prep_preprocessing.jpg">

Feature Description: <br>
* age: (numeric)
* job: type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
* housing: has housing loan? (categorical: 'no','yes','unknown')
* loan: has personal loan? (categorical: 'no','yes','unknown')
* balance: Balance of the individual.
* contact: contact communication type (categorical: 'cellular','telephone','unknown')
* month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
* campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
* pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; -1 means client was not previously contacted))
* poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
* deposit : as the client subscribed a term deposit? (binary: 1,0)


## Train-Test Split
To avoid data leakage from training data to test data, we only fit in training data <br>
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=44, stratify=y)`
 
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/initial_target_distribution.png">

There are 52.2% of No-subs customers and 47.8% of Subs customers, if the marketing campaign uses random calls

## Missing Value Handling
There are no columns with missing value

## Data Consistency and Anomalies
1. Grouping Common Labels<br>
  * `admin.` to `admin`
  * `other` to `unknown`

2. Drop Duplicate Data<br>
   Drop for train data, test data not dropped because not used for training the model <br>
  * drop duplicate in y (target):<br>
   `y_train.drop(X_train[X_train.duplicated()].index, inplace=True)`
  * drop duplicate in X (predictor):<br>
  `X_train.drop_duplicates(inplace=True)`

# Data Preprocessing
## Outliers Handling
### Equal Frequency Discretisation
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/equal_freq_bins.png">

Equal frequency discretisation divides the scope of possible values of the variable into N bins, 

> where **each bin carries the same amount of observations**. 

> This is **particularly useful for skewed variables** as it **spreads the observations over the different bins equally**. 

We find the interval boundaries by determining the quantiles.

Equal frequency discretisation 

> **using quantiles consists of dividing the continuous variable into N quantiles**, N to be defined by the user.

Equal frequency binning is straightforward to implement and by **spreading the values of the observations more evenly it may help boost the algorithm's performance**.

**Advantages**
* Handles outliers
* Good to combine with categorical encodings
* Improve value spread

**Limitation**
* Arbitrary binning may also disrupt the relationship with the target

## Categorical Encoding
### Rare Label Encoding
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/rare_label.png">

**Infrequent** labels are so few, that it is **hard to derive reliable information** from them. But more importantly, **infrequent labels tend to appear only on train set or only on the test set**:

* If only on the train set, they may cause over-fitting
* If only on the test set, our machine learning model will not know how to score them

**Grouping** infrequent labels or categories under a new category **called 'Rare' or 'Other' is the common practice** in machine learning for business.

* Grouping categories into rare for variables that show **low cardinality may or may not improve model performance**, **however**, we tend to re-group them into a new category to smooth model deployment.

* Grouping categories into rare for variables with **high cardinality**, **tends to improve model performance** as well.

### Ordinal Label Encoding
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/ordinal_label_enc.png">

**Ordering** the categories **according to the target means** assigning a number to the category from 1 to k, where k is the number of distinct categories in the variable, 

> but this numbering is informed by the mean of the target for each category.

**Advantages**
* Straightforward to implement
* Does not expand the feature space
* Creates monotonic relationship between categories
and target

**Limitations**
* May lead to over-fitting

## Model Benchmark
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/model_benchmark.jpg">

* Train the data with various model, then choose the model with the highest ROC-AUC
* The best model then selected for hyperparameter tuning

Light Gradient Boosting is choosed because of highest ROC-AUC

## Feature Selection
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/remove_multicoll.png">

### Remove Constant and Duplicated 
Remove low variance and duplicated data

### Remove Multicollinearity
Remove data which impact each other independent variables

> Two features were found, having a strong correlation (r>0.7), namely pdays and poutcome. Then the one with the higher variance will be selected.

### Boruta SHAP
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/boruta_shap.png">

Select top features using machine learning algorithm with SHAP importance

> Feature selected is based on Z-score compared to shadow feature. If lower than shadow feature z-score, it will be likely to unimportant

8 features were selected to be used for modeling with the condition that Z-score > shadow feature (random number)
* contact
* age
* Campaigns
* Job
* balance
* Housing
* month
* poutcome

# Model Building and Evaluation
## How LGBM Created
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/what_is_lgbm_1.jpg">

## What is LGBM?
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/what_is_lgbm_2.jpg">

Unlike other gradient boosting algorithms it uses level-wise tree growth. LightGBM uses an algorithm with leaf-wise tree growth. So when forming the same leaf level, LightGBM can reduce more loss and give better results and LightGBM also works very fast.

## Model Selection
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/model_selection.jpg">

### Hyperparameters Tuning
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/hyperparams_convergence.png">

The LGBM model will undergo hyperparameter tuning:
* n_estimators: number of boosting iterations
* max_depth: limit the max depth for tree model. This is used to deal with over-fitting when data is small. Tree still grows leaf-wise
* num_leaves: max number of leaves in one tree
* reg_alpha: L1 regularization to reduce overfitting
* path_smooth: helps prevent overfitting on leaves with few samples
* subsample: LightGBM will randomly select a subset of features on each iteration. For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree (can be used to speed up training and deal with overfitting)

The model ROC-AUC is convergence in 0.765

## Model Evaluation

<img src = "https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/roc_auc_standard.jpeg">

By these standards [ROC AUC rule of thumb from Hosmer and Lemeshow](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118548387), a model with an AUC score below 0.7 would be considered poor and anything higher would be considered acceptable or better.

<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/threshold%20selection.jpg">

* ROC is a probability curve and AUC represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. 

* Higher the AUC, the better the model is at predicting 0 classes as 0 and 1 classes as 1. By analogy, the Higher the AUC, the better the model is at distinguishing between patients with the disease and no disease. 

* It works by ranking the probabilities of prediction of the positive class label and calculating the Area under the ROC Curve which is plotted between True Positive Rates and False Positive Rates for each threshold value

There is no significant different between Train and Test ROC-AUC, so the model is not overfit

* Train ROC-AUC: 79%
* Test ROC-AUC: 77%

### Business Impact
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/business_impact.jpg">

* Based on the model, we might lose 33% potential users but we can filter customers who unlikely to subscribe up to 74%
    * In example, if we have 100 customers, the model can reject the 74 customers who unlikely to subscribe
* According to this [dataset](https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset), average marketing cost each person is 94 EUR
    > * If we use random call, the marketing cost will be `all_customers*94` = `2344*94` = `220366 EUR` 
    > * Of the 100 customers who are marketing targets, only 48 (48%) customers have the potential to subscribe
    
    > * If we use the model, the marketing cost will be `filtered_user*94` = `1061*94` = `99734 EUR`. In others word, the marketing cost `reduced by 54.7%` with 70% precision to correctly predict potential customers
    > * The model recommends 100 customers as marketing targets, then 70 people have the potential to subscribe
* If we use the model, we can `improve` subscribe ratio from `48% (random call) to 70%` with `reduced cost` up to `54.7%`

# Model Explainability
## Global Explainer - SHAP
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/shap_global_2.jpg">

## Local Explainer - SHAP
### Why Subscribe (1)?
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/subscribe.png">

* Not have housing loan (1)
* High balance (4)
* Contacted below 2 times during campaign(0)

### Why Not Subscribe (0)?
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/not_subscribe.png">

* Have housing loan (0)
* Low balance (0)
* Fail to subs in previous marketing campaign (1)

## Predictive Features

* Event = Subscribe
* Non-event = Not Subscribe

### poutcome
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/poutcome.png">
outcome of the previous marketing campaign

* Unknown / No Feedback: 0
* Failure: 1
* Success: 2 

> Customers who not subscribe (Failure) during previous campaign unlikely to subscribe

### month
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/month.png">
last contact month of year

* Mei: 0
* Juli: 1
* November: 2
* Juni: 3
* Agustus: 4
* Februari: 5
* April: 6
* Januari, Maret, September, Oktober, Desember: 7

> Customers who are contacted during Q1 and End of Year likely to subscribe

### contact
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/contact.png">
contact communication type

* Other: 0
* Telephone: 1
* Cellphone: 2

> Customers who are contacted via cellphone or have a cellphone likely to subscribe

### housing
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/housing.png">
has housing loan?

* Have housing loan: 0
* Do not have housing loan: 1

> Customers who do not have housing loans tend to subscribe

### balance
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/balance.png">
balance of the customers account

* balance < 67: 0
* 67-342: 1
* 342 - 860: 2
* 860-2157: 3
* balance > 2157: 4

> Customers who have a balance more than 67 EUR likely to subscribe. If want more strict the company can target the campaign for customers who have balance more than 860 EUR.

### age
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/age.png">
age of customers

* age < 31: 0
* 31-36: 1
* 36-42: 2
* 42-52: 3
* age > 52: 4

> Customers who are below 31 and above 52 likely to subscribe

### job
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/job.png">
type of job

* blue-collar: 0
* services: 1
* technician: 2
* admin: 3
* management: 4
* Other: 5
* retired': 6

> Customers who are works as blue-collar and services unlikely to subscribe

### campaign
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/campaign.png">
number of contacts performed during this campaign and for this client

* campaign < 2: 0
* 2-3: 1
* campaign > 3: 2

>Customers who are conctacted during campaign below 3 times likely to subscribe

# Conclusion and Recommendation
## Conclusion
* The LGBM model successfully classifies customers who are subscribed and non-subscribed and **not overfit**
    * **Train** ROC-AUC 79%
    * **Test** ROC-AUC 77%

* Model has **potential to lose 33% potential customers**, but model can **filter less potential customers** to subscribe up to **74%** (Specificity)

* Models can **reduce campaign costs** from **220366 EUR** to **99734 EUR**. In other words, **campaign costs reduced** by **54.7%** with **70% precision** to **predict** potential customers **correctly**.

* Successful model **increased subscribe rate** from **48%** (random calls) to **70%**

* There are 5 variables that **affect** customers to **subscribe**
    * Previous Campaign Outcomes
    * Campaign Month
    * ContactType
    * Housing Loans
    * Account Balances

## Recommendation
* **Adding more features** such as the interest rate offered, the form of promotion, income or level of education

* **Do a campaign** in **Q1 or in December** because it has a **high subscribe rate** and **avoid the campaign in May**

* **Target** customers **aged under 31** and **over 52** years **personally**

* **Avoid** customers who **didn't subscribe to the previous campaign** and **don't contact** the same person **more than 2 times** because it will **indicate spam**

* **Focus the campaign** on customers with an **account balance of more than 67 EUR** or if you want to be **stricter**, you can do it on customers with an **account balance of more than 860 EUR**
