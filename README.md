# **Bank Marketing Campaign Classification**
| **Report (PDF)** |
|:---:|
| **[Analysis](https://drive.google.com/file/d/1PX9So7JlmAs--c4fCEt6BYwxG5BZbUBk/view?usp=sharing)** |


# Table of Contents

[Business Understanding](#Business-Understanding)<br>
[Data Cleaning and Preparation](#Data-Cleaning-and-Preparation)<br>
[Exploratory Data Analysis](#Exploratory-Data-Analysis)<br>
1. [Product Price](#Product-Price)<br>
2. [Order Quantity Frequency](#Order-Quantity-Frequency)<br>
3. [Order by Category](#Order-by-Category)<br>
4. [Trends of Grand Total and Total Orders](#Trends-of-Grand-Total-and-Total-Orders)<br>
5. [Monthly Product Price by Order Status](#Monthly-Product-Price-by-Order-Status)<br>
6. [Monthly Discount Amount and Grand Total by Order Status](#Monthly-Discount-Amount-and-Grand-Total-by-Order-Status)<br>
7. [Retention Last 1 Year](#Retention-Last-1-Year)<br>
8. [RFM](#RFM)<br>

[Conclusion and Recommendation](#Conclusion-and-Recommendation)<br>

# Business Understanding
## Context
Based on [Investopedia](https://www.investopedia.com/terms/t/termdeposit.asp), a `term deposit` is a `fixed-term investment` that includes the deposit of money into an account at a financial institution. Term deposit investments usually carry `short-term maturities` ranging from `one month to a few years` and will have varying levels of required minimum deposits.

When an account holder `deposits` funds at a bank, the bank can `use that money` to `lend` to other consumers or businesses. `In return` for the right to use these funds for lending, they will `pay the depositor compensation` in the form of `interest` on the account balance. 

> A customer will `deposit or invest` in one of these accounts, agreeing `not to withdraw` their funds for a `fixed period in return` for a `higher rate of interest` paid on the account.

> The `interest earned` on a term deposit account is `slightly higher` than that paid on standard savings or interest-bearing checking accounts. The increased rate is `because access to the money is limited` for the `timeframe` of the term deposit.

> Term deposits are an `extremely safe investment` and are therefore very appealing to conservative, low-risk investors

For example
> * A lender (bank) may offer a 2% rate for term deposits with a two-year maturity. 
> * The `funds deposited` are then `structured as loans to borrowers` who are `charged 7% in interest` on those notes. 
> * This difference in rates means that the `bank makes a net 5% return`. 
> * The spread between the rate the bank pays its customers for deposits and the rate it charges its borrowers is called `net interest margin`. Net interest margin is a profitability metric for banks.

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


# Data Cleaning and Preparation
Pada tahap ini dilakukan pemahaman dan pembersihan data untuk mempersiapkan data sebelum dilakukan analisa <br>
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/data_preprocessing.jpg">

## Drop Unused Columns
Kolom-kolom berikut akan di drop karena tidak digunakan dalam analisa:<br>
`'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'increment_id',
 'sales_commission_code', 'Working Date', 'BI Status', ' MV ', 'Year', 'M-Y', 'FY', 'Month'`

Karena item_id unik pada setiap baris sedangkan sku tidak, maka item_id adalah ID transaksi/order bukan ID barang. 

Berikut adalah kolom yang akan digunakan dalam proses analisis:

* item_id: ID unik transaksi/order
* status: status barang yang dibeli 
* created_at: tanggal order
* sku: stock keeping unit, ID unik barang
* price: harga barang
* qty_ordered: jumlah barang yang dibeli
* grand_total: total harga barang yang dibeli 
* category_name_1: kategori barang
* discount_amount: diskon barang
* payment_method: cara pembayaran
* Customer Since: waktu pertama pelanggan menggunakan platform
* Customer ID: ID pelanggan

## Missing Value Handling
1. Drop Rows with All NaN Values<br>
`df.dropna(axis=0, how='all', inplace=True)`<br>
2. Drop Rows if Customer ID is NaN<br>
`df.dropna(subset=['Customer ID'], axis=0, inplace=True)`<br>
3. Most Frequent Imputation for NaN under 5%<br>
Karena NaN pada categorical data dibawah 5%, maka NaN akan di imputasi dengan nilai yang sering muncul.<br>
`df['sku'].fillna(df['sku'].mode()[0], inplace=True)`<br>
`df['status'].fillna(df['status'].mode()[0], inplace=True)`<br>
`df['category_name_1'].fillna(df['category_name_1'].mode()[0], inplace=True)`<br>

## Data Types Conversion
Convert tanggal ke datetime dan ID ke string <br>
`df['created_at'] = pd.to_datetime(df['created_at'])`<br>
`df['Customer Since'] = pd.to_datetime(df['Customer Since'])`<br>
`df['item_id'] = df['item_id'].astype('int').astype('str')`<br>
`df['Customer ID'] = df['Customer ID'].astype('int').astype('str')`<br>

## Data Consistency and Anomalies
1. Grouping Common Labels
 Beberapa label di kolom status memiliki kesamaan arti:

* RECEIVED dan CLOSED menjadi COMPLETE
* ORDER_REFUNDED dan REFUND menjadi CANCELED
* PENDING_PAYPAL, PAYMENT_REVIEW, PENDING, HOLDED, EXCHANGE, PAID, dan COD menjadi PROCESSING
* Label "PROCESSING" dan "FRAUD" sangatlah kecil (0.7 dan 0.001%), jadi bisa di drop jika tidak diperlukan dalam analisis 
* EASYPAY_MA dan EASYPAY_VOUCHER menjadi EASYPAY

2. Drop Anomalies Data
Terdapat anomali data pada kolom "status" dan "category_name_1" dengan label "\\N"
* Karena persentasenya sangat kecil (1%), maka baris dengan label "\\N" akan di drop

3. Replace Anomalies Data
Terdapat anomali data pada kolom "grand_total" dan "discount_amount", yaitu nilainya ada yang negatif (harga dan diskon seharusnya tidak negatif)
* Maka akan di filter hanya include grand total dan diskon yang lebih dari 0

## Add New Features (Feature Improvements)
1. Order Quantity Binning <br>
Group order quantity menjadi beberapa label seperi 6-10,11-20,dst <br>
`df['qty_bins'] = pd.cut(`<br>
    `df.qty_ordered,`<br>
    `bins=[0, 1, 2, 3, 4, 5, 10, 20, 100, 1000],`<br>
    `include_lowest=True,`<br>
    `labels=["1", "2", "3", "4", "5", "6-10", "11-20", "21-100", "101-1000"]`<br>
`)`<br>

2. Monthly Period <br>
Buat fitur baru berdasarkan orde bulanan <br>
`df["order_month"] = df['created_at'].dt.to_period("M")` <br>

3. Quarter Period <br>
Buat fitur baru berdasarkan orde kuartal <br>
`df["order_quarter"] = df['created_at'].dt.to_period("Q")` <br>

4. Day of Week <br>
Buat fitur baru berdasarkan orde day of week <br>
`dow_mapping = {`<br>
    `0: "Monday",`<br>
    `1: "Tuesday",`<br>
    `2: "Wednesday",`<br>
    `3: "Thursday",`<br>
    `4: "Friday",`<br>
    `5: "Saturday",`<br>
    `6: "Sunday",`<br>
`}`<br>
`df["day_of_week"] = df['created_at'].dt.dayofweek`<br>
`df["day_of_week"] = df["day_of_week"].map(dow_mapping)` <br>

5. Is Complete Order? <br>
Buat fitur baru sebagai indikator apakah customer menyelesaikan order atau tidak <br>
`df["is_complete"] = df['status'].apply(lambda x: 1 if x in ["COMPLETE"] else 0)` <br>

## Check/Drop Duplicate Data <br>
`df.duplicated().sum()`<br>
Tidak ditemukan data duplikat

# Exploratory Data Analysis
## Product Price

Untuk menjawab pertanyaan berikut:
* Berapa rentang harga produk yang sering dibeli customer?
* Produk apa yang berada pada rentang harga tersebut?

Akan dilakukan analisa terhadap harga dan kategori produk <br>
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/analysis1.jpg">

## Order Quantity Frequency

Untuk menjawab pertanyaan berikut:
* Berapa order quantity customer?
* Apa pengaruh order quantity terhadap status order customer?

Akan dilakukan analisa order quantity berdasarkan status ordernya<br>
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/analysis2.jpg">

## Order by Category

Untuk menjawab pertanyaan berikut:
* Apakah ada perbedaan total order dan complete order rate tiap kategori

Akan dilakukan analisa total order dan complete rate berdasarkan produk catogry<br>
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/analysis3.jpg">

## Trends of Grand Total and Total Orders

Untuk menjawab pertanyaan berikut:
* Bagaimana pola grand total dan total order dari waktu ke waktu?
* Apa dampaknya terhadap complete order rate

Akan dilakukan analisa grand total (setelah diskon) tiap order secara quarter dan bulanan beserta rasio complete order<br>
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/analysis4.jpg">
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/analysis5.jpg">

## Monthly Product Price by Order Status

Untuk menjawab pertanyaan berikut:
* Apakah tidak ada perbedaan distribusi customer dengan Complete Order dan Canceled Order?

Akan dilakukan analisa terhadap rata-rata harga bulanan berdasarkan status ordernya<br>
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/analysis6.jpg">

## Monthly Discount Amount and Grand Total by Order Status

Untuk menjawab pertanyaan berikut:
* Apakah ada pengaruh jumlah diskon terhadap complete dan cancel order?
* Apakah ada pengaruh grand total terhadap complete dan cancel order?

Akan dilakukan analisa jumlah diskon dan grand total berdasarkan status ordernya<br>
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/analysis7.jpg">

## Retention Last 1 Year

Untuk menjawab pertanyaan berikut:
* Apakah customer masih menggunakan platform e-commerce dalam 1 tahun kebelakang?
* Apakah ada pola tertentu pada aktivitas customer dalam 1 tahun?

Analisa Retention dilakukan dengan mengecualikan order yang fraud<br>
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/analysis8.jpg">

## RFM

Untuk menjawab pertanyaan berikut:
* Bagaimana distribusi recency, frequency order dan monetary value tiap customer?

Analisa RFM dilakukan dengan mengecualikan order yang fraud

[RFM segmentation references](https://github.com/daniel-isidro/customer_segmentation)

NB: Average Order Value tidak diterapkan karena distribusi price tidak normal sehingga missleading dalam perancangan strategi. Contohnya AOV tinggi dengan price tinggi tapi order rendah, sekilas menunjukan performance yang bagus tetapi jika ditinjau secara quantity, akan menunjukan quantity penjualan yang rendah. <br>
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/analysis9.jpg">
<img src="https://github.com/mch-fauzy/purwadhika-capstone-project-module-2/blob/main/media/analysis10.jpg">

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
