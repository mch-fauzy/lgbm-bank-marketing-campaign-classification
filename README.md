# **Bank Marketing Campaign Classification**
| **Report (PDF)** |
|:---:|
| **[Analysis](https://drive.google.com/file/d/1PX9So7JlmAs--c4fCEt6BYwxG5BZbUBk/view?usp=sharing)** |


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
### poutcome
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-3/blob/main/media/poutcome.png">
outcome of the previous marketing campaign
* Unknown / No Feedback: 0
* Failure: 1
* Success: 2 

> Customers who not subscribe (Failure) during previous campaign unlikely to subscribe
> 
### month
### contact
### housing
### balance
### age
### job
### campaign


# Conclusion and Recommendation
## Conclusion
Dari analisis yang telah dilakukan, dapat disimpulkan:
* Customer lebih sering membeli produk dengan harga di bawah 939.
* Lebih dari 80% order memiliki 1 quantity per order dengan rata-rata persentase complete order cenderung diatas 55% pada rentang order quantity 1-3 dan 6-10. Customer lebih memilih untuk melakukan pembelian yang lebih kecil dengan produk yang lebih sedikit per order.
* Analisis menunjukkan bahwa Men's Fashion dan Mobiles & Tablets adalah kategori tertinggi yang diorder customers, terhitung sekitar 15-20% dari total transaksi. Namun, mereka menunjukkan tingkat complete order rate yang berbeda. Rata-rata complete order rate untuk Mobiles & Tablets (42%) adalah sekitar 15% lebih rendah daripada Men's Fashion (57%).
* Rata-rata grand total terus meningkat sejak 2017 Q3. Bahkan, meningkat 238% dari Q3 2017 ke Q3 2018 dengan total orders tertinggi terjadi pada Q4. Namun, selama periode waktu yang sama, complete order rate mengalami penurunan dari waktu ke waktu tanpa pola yang jelas. 
* Total orders terbesar terjadi pada bulan November dengan order lebih dari 60k. Total orders kedua terbesar terjadi pada bulan Mei dengan order mencapai 30k.
* Rata-rata grand_total stabil dalam skala harian. Namun, total order tertinggi terjadi pada hari Jumat. Selain itu, tampaknya complete order rate tetap stabil dan sebagian besar di atas 50%.
* Customer dengan canceled order memiliki median rata-rata bulanan price 3x lebih tinggi daripada customer dengan complete order. Hal ini mungkin menunjukkan bahwa harga yang tinggi merupakan faktor yang menyebabkan customer mengurungkan niat untuk membeli sehingga terjadi pembatalan order.
* Analisis menunjukkan bahwa order yang dibatalkan memiliki median rata-rata grand total 3x lebih tinggi daripada order yang berhasil sedangkan  diskon secara statistik tidak berdampak signifikan pada complete order rate. Ada kemungkinan bahwa customer membatalkan atau refund order karena produk yang sebenarnya berbeda dari yang diiklankan
* Retention rate analysis menunjukkan bahwa tingkat retention customers dalam 1 tahun secara keseluruhan tergolong rendah. Hanya rata-rata 5.5% customer melakukan repeat order dalam 90 hari sejak pembelian terakhir mereka. Berdasarkan [Harvard Business School](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)  mendapatkan customers baru lebih mahal 5-25 kali daripada mempertahankan customers yang ada
* 50% customers masih aktif dalam melakukan transaksi yaitu di segmen Loyal, Potential, New, dan Champions
* Segmen prioritas untuk high recency (R 3-4) adalah Champions, Loyal, Potential, dan New
* Segmen prioritas untuk low recency (R 1-2) adalah At Risk, About to Sleep, dan Can't Lose Them

## Recommendation
1. Perusahaan disarankan fokus untuk menawarkan lebih banyak produk dalam kisaran harga dibawah 939 seperti "Men's Fashion", "Home & Living", dan "Health & Sports" untuk memenuhi permintaan customer dan meningkatkan kinerja penjualan.
    * Perlu analisa lebih lanjut untuk produk dengan harga diatas 939, karena beberapa produk memiliki rentang harga yang lebar dan adanya brand yang mengeluarkan produk low-budget.
2. Minimalisir pemberian diskon karena kurang berdampak pada complete order rate
3. Perusahaan direkomendasikan berfokus pada promosi penawaran bundel yang lebih kecil, order quantity 1-3  atau menerapkan loyalty program untuk customer yang melakukan order dengan quantity 6-10 untuk memaksimalkan kesempatan complete order demi meningkatkan penjualan.
    * Jika perusahaan ingin lebih berhati-hati dalam menerapkan strategi promosi, lebih baik berfokus pada order quantity 1-5 dan 11-20 karena secara distribusi memiliki median persentase complete order lebih stabil diatas 55% dan rentang yang lebih sempit.
4. Perusahaan dapat fokus mempersiapkan volume order yang tinggi di bulan November dan Mei serta mempertimbangkan untuk meningkatkan stock pada bulan tersebut untuk memenuhi permintaan customer
5. Perusahaan bisa mempertimbangkan untuk menyediakan program cicilan atau memastikan bahwa produk aktual yang dikirim ke customer sesuai dengan iklan untuk meminimalkan jumlah order yang dibatalkan.
6. Contoh strategi yang dapat digunakan antara lain rekomendasi produk berdasarkan pembelian sebelumnya, loyalty program, cross-selling, upselling, re-engagement emails, follow-up phone calls atau surveys untuk memahami mengapa mereka belum melakukan pembelian akhir-akhir ini. 
    * Prioritaskan pada segmen Champion, Loyal, Potential, dan New untuk meminimalisir risiko churn
    * Prioritaskan pada segmen At Risk, About to Sleep, dan Can't Lose Them untuk reactivation customers

### Potential Opportunity
* Soghaat memiliki complete order rate tertinggi tetapi total ordernya rendah. Soghaat adalah produk makanan manis, complete order rate tinggi menunjukkan bahwa customer puas dengan produk dan layanannya. Namun, penting untuk berfokus pada peningkatan total order untuk kategori ini.
* Pada bulan Juni rata-rata grand total cenderung lebih tinggi dengan rentang mencapai 1.5-1.7 kali rata-rata grand total bulan-bulan lainnya. Perusahaan dapat memanfaatkan tren rata-rata grand total yang lebih tinggi di bulan Juni dengan memberikan penawaran dan promosi untuk menarik lebih banyak customer.
* Customers dengan frekuensi order tinggi (F 3-4), cenderung memiliki monetary value tinggi (M 3-4) meskipun sudah jarang aktif. Perusahaan bisa melakukan pendekatan pada customer yang jarang aktif (R 1-2) tetapi memiliki FM tinggi untuk dapat meningkatkan penjualan
