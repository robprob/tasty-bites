# Tasty Bites: Analysis and Machine Learning Solution
This [project](https://github.com/robprob/tasty-bites/blob/main/Tasty-Bites.ipynb) was completed for [certification](https://github.com/robprob/tasty-bites/blob/main/Professional-Data-Scientist-Certification.pdf) as a Professional Data Scientist through DataCamp. The assignment simulated a real-world data job where I was provided data from a recipe website, "Tasty Bites", with the request of correctly predicting popular recipes that would lead to increased revenue. In accordance with the requests of the customer, I performed advanced data validation, cleaning, exploratory analysis, and created a predictive machine learning model to deploy into production. Most notable about this project was my handling of "advanced missigness" across 4 separate columns, determining the statistical significance of the missing data and whether entire removal of such rows would have a negative effect on the ability of other features to accurately predict the target variable.. I then presented my findings and the trained model to the customer. Meeting and exceeding their expectations, I discussed future plans and recommendations to improve the model and identified other financially beneficial work I could help them with. The slides that accompanied my presentation can be found [here](https://github.com/robprob/tasty-bites/blob/main/Tasty-Bites-Presentation.pdf). I found this presentation most beneficial in improving my ability to succintly relay my work to a non-technical audience while keeping all of the important information. As this project was for a certification exam, the notebook contains very detailed notes in markdown, but I will provide some of the most important points below as a summary:


## Data Validation
The provided dataset had 947 entries across 8 columns, 5 of which columns contained some missing values. After validation, 892 entries and 8 columns remain.

1. recipe: validated all 947 entries as unique identifiers
2. calories: logically validated using descriptive statistics, removed 52 entries with missing values
3. carbohydrate: logically validated using descriptive statistics, removed 52 entries with missing values
4. sugar: logically validated using descriptive statistics, removed 52 entries with missing values
5. protein: logically validated using descriptive statistics, removed 52 entries with missing values
6. category: consolidated 'Chicken Breast' with 'Chicken' and converted to 'category' data type, resulting in the 10 provided possible groupings
7. servings: removed 3 entries with ambiguous interpretability, converted to int64 and logically validated the remaining values
8. high_traffic: validated target variable as binary and encoded as Boolean (NaN = False, 'High' = True)


## Calories, Carbohydrate, Sugar, Protein (Missingness)
There are 52 entries where the numeric values are completely missing across all 4 columns: `calories`, `carbohydrate`, `sugar`, and `protein`.

These missing entries make up more than 5% of the total data and initially did not appear to be missing completely at random (MCAR); a statistically significant difference in proportions of recipes that were `high_traffic` was substantiated using a Welch's t-test between recipes that were missing vs. not-missing values in these 4 numeric columns. 

However, the only additional feature information lost from dropping these would be from `category` and `servings` columns (`recipe` is unique identifier column and `high_traffic` is target variable). To justify this, I evaluated the relationship between missingness of numeric data within the remaining columns with respect to the target variable.

### Within-Category Missingness
The DataFrame below, missing_bycat, displays the results of my function evaluate_missing('category') by category:
- `pct_missing`: percentage of total values missing (within category)
- `pct_high`: percentage of recipes that resulted in 'high_traffic'
- `pct_missing_high`: percentage with missing numeric values that resulted in 'high_traffic'
- `pct_nonmissing_high`: percentage with non-missing numeric values that resulted in 'high_traffic'

Results: missing_bycat:
- Some food groupings appear to have a higher frequency of missing numeric data, with 'Pork' being the highest, at 13.10%. Two categories are not missing any numeric data at all ('Beverages', 'Breakfast').
- The proportion of recipes that are `high_traffic` vary between categories.
- While the proportion of recipes that result in `high_traffic` appear to vary when numeric values are missing, these small sample sizes do not appear to exert enough influence to manifest a significant difference between the total catagory-specific `high_traffic` proportion and the `high_traffic` proportion for entries in that category _without_ missing numeric data. This is substantiated using 1samp t-testing.


This has been repeated for the other remaining column, 'servings', that has similar results without any initially notable differences.

![image](https://github.com/user-attachments/assets/3c6d434b-138b-4d0d-b050-91646d30d292)

### Hypothesis Testing
Null Hypothesis: 
- `high_traffic` proportion of non-missing entries ('pct_nonmissing_high' is sample prop) is not significantly different from the `high_traffic` proportion for all of the entries in that category, missing or not ('pct_high' is population prop).

Alternative Hypothesis:
- `high_traffic` proportion of non-missing entries ('pct_nonmissing_high' is sample prop) is significantly different from the `high_traffic` proportion for all of the entries in that category, missing or not ('pct_high' is population prop).

None of the p-values from the 1-sample t-tests are below the significance level, alpha, indicating that there is not enough evidence in the difference between the population proportion and sample proportion to attribute these differences to anything but chance alone. This further substantiates the evidence obtained earlier that individual feature importance of `category` and `servings` will not be significantly alterred or misrepresented by dropping the entries that have missing numeric data in `calories`, `carbohydrate`, `sugar`, and `protein`.

### Final Thoughts on Missingness
While this my best solution, it should be emphasized that this is still not completely optimal. Between groups, there appears to be variance in the proportion of missing values, as discussed earlier, which could result in some misrepresentation of the numeric data columns `calories`, `carbohydrate`, `sugar`, and `protein` if they prove to be of importance, but the values of these columns also greatly vary, both within and between groups.

This is displayed in the DataFrame below, cals_by_cat, which provides descriptive statistics for `calories` grouped by `category`.

For these reasons, I will not opt for imputation by a constant, whether by single or multiple categories (mean/mode grouped by both `category` and `servings`), and have instead chosen to drop the rows with missing values.

## Exploratory Analysis
![image](https://github.com/user-attachments/assets/498a6e25-d66b-453f-9d69-3f7289fa224f)
![image](https://github.com/user-attachments/assets/c9d5c18e-27ad-416f-9b58-c468bb68c165)
![image](https://github.com/user-attachments/assets/5beb64d5-d330-40bd-9fe5-c6e5ba3aa561)
![image](https://github.com/user-attachments/assets/5b031003-2586-4c87-8847-c1698854a887)

## Model Development
Predicting whether a recipe will lead to `high_traffic` is a binary classification, supervised learning problem.

Evaluation metrics
- **accuracy_score**: Baseline accuracy. May not be best evaluator with class imbalance of ~ 3/2 (True/False).
- **recall**: Tasty Bites asked if we can "correctly predict high traffic recipes 80% of the time", also known as a sensitivity/recall of 80%.
- **roc_auc_score** to quantify True Positive vs. False Positive rate at various thresholds of classification.

These performance metrics should be suitable for evaluating a binary classifier using this dataset with similarly-sized binary outcomes and ensure the results are not due to chance alone.
![image](https://github.com/user-attachments/assets/c0d3db18-8df9-4f92-8682-32f031eaf7cf)
![image](https://github.com/user-attachments/assets/fb1fad60-f909-41ff-af22-5f57ade97168)

## Business Metrics
The customer asked if we could "correctly predict high traffic recipes 80% of the time". In the test set, the Random Forest Classifier correctly predicted 131 out of 160 `high_traffic`, resulting in a recall of 81.9%.

This is a helpful KPI to evaluate current and future iterations of this model. While keeping other metrics balanced, improving recall will reduce the proportion of false negative results and help to more consistently identify recipes that will increase website traffic and subscription purchases.

## Recommendations
This model exceeds the requests of the customer, and can be deployed immediately to help identify which recipes will result in `high_traffic` when featured. This should already provide significant benefit through featuring daily recipes that more consistently cause `high_traffic` and a resulting increase in subscription purchases. However, while testing this model in-field, it would be most beneficial to continue improving this model by asking some questions and gathering more data. Below is my plan moving forward:

Deployment
- One-off deployment should be very cost-effective until this model is updated. This model only has to be deployed on and utilized via customer's device(s) containing the website data.
- I will create a practical dashboard application with a simple user interface that allows the customer to pull new website/recipe data and view model predictions in the format they'd like, including visually pleasing and easily interpretable graphics.  

Collecting Additional Data
- Missing data: 52 entries are missing 4 columns of data with high feature importance. Why is this data missing? Can it be corrected?
- 'Time to make', 'Cost per serving', 'Ingredients': This information is displayed on the provided website example but was not included in the dataset. Perhaps it is relevant?

Feature Engineering
- `category`: I used the 10 food groupings that were provided, as the customer may deem it important that they be kept as specified. However, some groupings are not clear; it is unknown whether 'Chicken' and 'Pork' can be grouped into the 'Meat' category when nutritional metrics vary across these food groupings. Additionally, a recipe can only belong to one food grouping, but these groupings are not exclusive. For example, a recipe for chicken salad could potentially belong to 'Chicken', 'Vegetable', 'Meat', 'One dish meal', 'Lunch/Snacks', or more. This column should have likely been split into multiple features, such as meal-type (Breakfast, Lunch, Dinner) or a list of main ingredients, so each recipe is not limited to one, vague food grouping when no other qualitative information is available. 
- `high_traffic`: In the data provided, more than 60% of recipes lead to `high_traffic` (574/947). Perhaps the threshold of what it means to increase website traffic should be changed, so relevant features are more readily idenfitied before being generalized. Another option is to obtain more data to change to a regression problem, identifying how much a given recipe might increase traffic (customer insinuated they can quantify this increase in traffic). After all, it is more beneficial to choose recipes that increase traffic "by as much as 40%" rather than recipes that will produce a marginally smaller increase.

Improving the model
- Once more data is collected, can test other models to see if new data is a better fit for a different base model
- Perform an exhaustive GridSearchCV (instead of Random)
- Improve model stability using BaggingClassifier and OOB evaluation
