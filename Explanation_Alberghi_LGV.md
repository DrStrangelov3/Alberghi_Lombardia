
## Class prediction of lodging structures in Lombardy

This model aims to predict the type of lodging facilities located in Lombardy; a region of Italy. This is an example of a supervised classification problem where there are 5 different classes to predict: `B&B`, `Case_Appartamenti` , `1_a_3_Stelle` , `4_a_5_Stelle` , `Campeggio` and a mix of numerical and categorical data to do so. 
<br>
>- `ID`: Index
>- `PROVINCIA`: Letter code of the county capital 
>- `COMUNE`: City in which the structure is located
>- `LOCALITA`: Name of the sub-locality if available
>- `CAMERE`: Number of rooms
>- `SUITE`: Number of suites if available
>- `LETTI`: Number of beds
>- `BAGNI`: Number of bathrooms
>- `PRIMA_COLAZIONE`: Breakfast included
>- `IN_ABITATO`: 
>- `SUL_LAGO`: Lakefront 
>- `VICINO_ELIPORTO`: Near heliport
>- `VICINO_AEREOPORTO`: Near Airport
>- `ZONA_CENTRALE`: Central
>- `VICINO_IMP_RISALITA`: Near ski lifts
>- `ZONA_PERIFERICA`: Outskirt area
>- `ZONA_STAZIONE_FS`: Near train station
>- `ATTREZZATURE_VARIE`: Misc. features 
>- `CARTE_ACCETTATE`: Credit cards accepted 
>- `SPORT`: Sport facilities
>- `CONGRESSI`: Convention halls 
>- `LATITUDINE`: Lat.
>- `LONGITUDINE`: Long.
>- `OUTPUT`: Target classes

The dataset has a shape of 6775 rows and 25 columns 



### Challenges

Before we can fit a ML model, this dataset presents a few challenges:

1. High number of missing values `NaN`, `data.isna().sum().sum()` 30938
2. Transform objects to usable features
3. Classes imbalance
4. Features elimination

## Data Cleaning and Features Extraction

In this section I will explain my thought process behind my data cleaning decisions. This process is aimed to solve some of the challenges listed above; especially number 1 and 2. As I did in my code, I will approach each feature individually starting from the first. 

**ID**: In our dataframe this column is superfluous as we have a built-in index. This feature was dropped  `data.drop(['ID'], axis=1)`

__PROVINCIA__:  This column was dropped from the dataset. However, part of this feature was used by the external data.

__COMUNE__: I also dropped this column, but it was used as key to merge external data.

__LOCALITA__: As most of this feature is composed by `NaN`, I transformed it to a dummy variable on whether the data was present or not. This feature could be significant to discriminate if a structure is in a rural area. `data['LOCALITA'].apply(lambda x: 0 if pd.isnull(x) else 1)`

__CAMERE__, __SUITE__, __LETTI__, __BAGNI__, __PRIMA_COLAZIONE__: Those features where left untouched as there was no missing value.

__IN_ABITATO__, __SUL_LAGO__, __VICINO_AEREOPORTO__, __ZONA_CENTRALE__, __VICINO_IMP_RISALITA__, __ZONA_PERIFERICA__, __ZONA_STAZIONE_FS__: The same approach was used on all those features as they are missing the same 107 entries. All, except one, belong to `Case_Appartamenti`. My idea is that a home owner is less likely to fill the facility's characteristics field. This seems to follow a pattern for this category across the dataset. My solution is to fill `NaN` with 0. `data['ZONA_PERIFERICA'].fillna(0, inplace=True)`

__VICINO_ELIPORTO__: Since there are only 4 facilities that have this feature, I decided to drop the column.

__ATTREZZATURE_VARIE__: This feature, when present, is a long string of text listing all the amenities of the establishment. <font face=Courier color=#DC143C>"Ristorante, Riscaldamento, SKY"</font>. After splitting the string, since the amenities are many but standard (probably selected from a menu), I counted how many times each one appeared. I then arbitrarily selected only those that would appear 75 times or more and made them into dummy variables. This added 54 new columns to the original dataset. It seems that the most common amenity is <font face=Courier color=#DC143C>"Accettazione animali domestici"</font> appearing 2165 times.

__CARTE_ACCETTATE__: This feature as well, when present, is a string of text which lists all the cards accepted in the establishment. However, sometimes specific brands are listed and others: <font face=Courier color=#DC143C>"Tutte"</font> (all of them) is present. I converted the feature to a dummy whether cards where reported or not. `data[['CARTE_ACCETTATE']].apply(lambda x: 0 if pd.isnull(x[0]) else 1, axis=1)`

__LINGUE_PARLATE__: In this case, I assumed that business driven facilities such as hotels would be more likely to report all the languages spoken by their employees. I also assumed that higher tier establishments would be more incentivized to have a polyglot staff. I converted to numeric by counting how many languages were spoken and filled the `NaN` with 0. `data['LINGUE_PARLATE'].apply(lambda x: len(x) if x[0] != '-' else 0 )`

__SPORT__: My approach to this feature has been similar to the one used for ATTREZZATURE_VARIE. However, in this case, I grouped together similar sports and discarded the rest. I combined sports with a field, summer sports, winter sports, indoor sports and swimming pool (combining indoor and outdoor). I kept the gym feature `['Fitness/centro salute']` as is. Those were made into dummies and merged to the original dataset. 

__CONGRESSI__: This feature, when present, is a string of text including the number of convention halls, min capacity and max capacity. The beast option would have been to extract the max capacity, but a few entries were missing this data. I opted to extract only the number of halls which was available for all non-`NaN` values. The rest was filled with 0. This was done with a simple regex `data['CONGRESSI'].fillna('0').apply(lambda x: re.search(r'(\d+){1}', x)[0]).astype(int)`

__LATITUDINE__, __LONGITUDINE__: Those features were only used for plotting.

__OUTPUT__: Classes where converted to numeric on a scale 1:5. `output = {'Case_Appartamenti':1, 'Campeggio':2, 'B&B':3, '1_a_3_Stelle':4, '4_a_5_Stelle':5 }`

### External Data

To improve the predictive power of the model I used an external dataset which includes demographic and geographic data of every city in Italy in 2017. The data were collected by ISTAT and it is available on http://ckan.ancitel.it/dataset/. From this dataset I used only 5 features: 
<br>
>- `ZonaClimatica`: Climate 

>- `DensitaDemografica`: Pop. density 

>- `AltezzaCentro`: Altitude 

>- `TipoComune`: County capital [0,1]

>- `GradoUrbaniz`: Urban level [Alto, Medio, Basso]

<br>
The final dataset has a shape of 6775 rows and 80 columns

## Modeling

#### Model selection

The models selected are meant to overcome the last two challenges: classes imbalance and features elimination. The simplest models to apply in our situation, without rebalancing our classes, are decision trees and therefore: ensemble methods. For our dataset I decided to use a bagging and a boosting ensemble; RandomForestClassifier and XGBClassifier. 


#### Preprocessing

While our models mostly bypass our challenges of classes imbalance and feature pruning, I tried applying some dimensionality reductions techniques. First I rescaled the dataset with `MinMaxScaler()`. I also tried to apply `PCA()` it didn't provide any significant score improvement. Because of this, I commented this part of code out to mantain features interpretability.
Last I tried a recursive feature elimination method `RFECV()` which only slightly improved my best accuracy score.

#### Overfitting prevention

Since the models selected are prone to overfit, it is essential to find a way to counter it. Therefore, I splitted the data into a train and a test set `train_test_split(X_scaled, Y, test_size=0.25, random_state=42)` this way I can train my model on 75% of my data and then test it on the remaining 25%. Also, while fitting the model I applied a 5 folds Cross Validation which increases the computational intensity but prevents overfitting. The last confirmation that my models did not overfit is that my cv `best_score_`  and my `accuracy_score` on the test set are extremely close. 

#### Hyperparameter optimization

In order to optimize my models I built a pipeline and used `GridSearchCV()` to find the best hyperparameter. With a process of trial and error, I searched over filds like `n_estimators`, `max_depth`, `reg_alpha`, `reg_lambda`, `criterion`, and `learning_rate`.

#### Score and Performance

Overall both model performed above expectations. On the test set:
<br>
>- `RandomForestClassifier()` has an `accuracy_score` = 0.893

>- `XGBClassifier()` has an `accuracy_score` = __0.9114__ 

By looking at the confusion matrix of the models, they have similarl performance. The biggest obstacle seems to be the misclassification between `Case_Appartamenti` and `B&B` and vice versa. Also the models seem to misclassify `4_a_5_Stelle` into `1_a_3_Stelle`. These errors are understandable as characteristics are very similar. Overall, the model is able to accurately distinguish between the macro classes of hotels and non-hotels.

#### Future improvments

I believe there could be some improvements to be done in the future. Additional data could be gathered trough web scraping. This would both increase our dataset and provide new features like room price and quality of nearby shops and venues. Also I would be interested in deploying a neural network with Keras and see if that would improve our best score. 
