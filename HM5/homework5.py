# classification_pipeline_customer.py
# Requires: pandas, numpy, scikit-learn, matplotlib, scipy
# pip install pandas numpy scikit-learn matplotlib scipy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    cohen_kappa_score, roc_auc_score
)
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
RANDOM_STATE = 42

df = pd.read_csv("CustomerPurchasingBehavior.csv")  # ensure file is in working dir

# -----------------------------
# Specifying a target in target_col
#    - ASSUMPTION: aggregate purchase_amount into 3 classes (Low/Med/High)
# -----------------------------
target_col = 'purchase_amount'
df['target'] = pd.qcut(df[target_col], q=3, labels=['low', 'medium', 'high'])

# Preprocessing (specifying the features and the target)
X = df.drop(columns=['user_id', 'target'])
y = df['target']

# define numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# Preprocessor operations
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Models + hyperparameter distributions for RandomizedSearchCV
models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        'params': {
            'clf__C': np.logspace(-3, 3, 20),
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE),
        'params': {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [None, 5, 10, 20],
            'clf__min_samples_split': [2, 5, 10]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'clf__n_estimators': [50, 100, 200],
            'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'clf__max_depth': [3, 5, 7]
        }
    },
    'SVC': {
        'model': SVC(probability=True, random_state=RANDOM_STATE),
        'params': {
            'clf__C': [0.1, 1, 10, 100],
            'clf__gamma': ['scale', 'auto']
        }
    }
}

# Evaluate with 30 random subsampling runs
n_runs = 30
test_size = 0.3
random_states = list(range(RANDOM_STATE, RANDOM_STATE + n_runs))

results = {}
for name, entry in models.items():
    print(f"\nEvaluating model: {name}")
    acc_list = []
    f1_list = []
    kappa_list = []
    auc_list = []
    conf_mat_sum = None
    # store predictions per run for Wilcoxon comparison
    per_run_predictions = []  # predictions/labels on test set per run
    per_run_true = []   # true labels per run
    for i, rs in enumerate(random_states):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=rs)

        # assemble pipeline
        pipe = Pipeline([
            ('preproc', preprocessor),
            ('select', SelectKBest(mutual_info_classif, k='all')),  # keep all by default; can be tuned
            ('clf', entry['model'])
        ])

        # hyperparameter tuning on training set - RandomizedSearchCV
        param_dist = entry['params']
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
        rnd_search = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=10, scoring='accuracy',
            n_jobs=-1, cv=cv_inner, random_state=rs, verbose=0
        )
        rnd_search.fit(X_train, y_train)
        best = rnd_search.best_estimator_

        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        kappa = cohen_kappa_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        conf_mat_sum = cm if conf_mat_sum is None else conf_mat_sum + cm

        # ROC AUC: need probability estimates and binarized y for multiclass
        try:
            y_proba = best.predict_proba(X_test)
            classes = best.classes_

            # Binarize true labels
            y_test_b = label_binarize(y_test, classes=classes)

            # if y_test_b has shape (n_samples, n_classes)
            auc = roc_auc_score(y_test_b, y_proba, average='macro', multi_class='ovr')
        except Exception as e:
            auc = np.nan

        acc_list.append(acc)
        f1_list.append(f1)
        kappa_list.append(kappa)
        auc_list.append(auc)
        per_run_predictions.append(y_pred)
        per_run_true.append(y_test.values)

    results[name] = {
        'accuracy_runs': np.array(acc_list),
        'f1_runs': np.array(f1_list),
        'kappa_runs': np.array(kappa_list),
        'auc_runs': np.array(auc_list),
        'confusion_sum': conf_mat_sum,
        'preds': per_run_predictions,
        'trues': per_run_true,
        'mean_accuracy': np.mean(acc_list),
        'std_accuracy': np.std(acc_list),
        'mean_f1': np.mean(f1_list),
        'std_f1': np.std(f1_list),
        'mean_kappa': np.mean(kappa_list),
        'std_kappa': np.std(kappa_list),
        'mean_auc': np.nanmean(auc_list) if np.any(~np.isnan(auc_list)) else np.nan,
        'std_auc': np.nanstd([v for v in auc_list if not np.isnan(v)]) if np.any(~np.isnan(auc_list)) else np.nan
    }

print("\n\n===== SUMMARY =====")
for name, res in results.items():
    print(f"\nModel: {name}")
    print(f"  Mean accuracy: {res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}")
    print(f"  Mean F1 (macro): {res['mean_f1']:.4f} ± {res['std_f1']:.4f}")
    print(f"  Mean Cohen's kappa: {res['mean_kappa']:.4f} ± {res['std_kappa']:.4f}")
    print(f"  Mean AUC (macro, OVR): {res['mean_auc']:.4f} ± {res['std_auc']:.4f}")
    print(f"  Aggregated confusion matrix (rows=true, cols=pred):\n{res['confusion_sum']}")

# Wilcoxon test between two best models - by mean accuracy
ordered = sorted(results.items(), key=lambda kv: kv[1]['mean_accuracy'], reverse=True)
if len(ordered) >= 2:
    name1, res1 = ordered[0]
    name2, res2 = ordered[1]
    print(f"\nTop two models: {name1} ({res1['mean_accuracy']:.4f}), {name2} ({res2['mean_accuracy']:.4f})")

    # For Wilcoxon we need paired samples: accuracy per run aligned
    # We'll use the accuracies arrays, they are paired by iteration index
    accuracy1 = res1['accuracy_runs']
    accuracy2 = res2['accuracy_runs']

    # Check for ties and zero differences - wilcoxon requires at least some non-zero differences
    try:
        # ignore the warning, the properties exist, documentation at :
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
        # additionally, I checked the debugger and the values are present
        # I will print them just to remove the time to run it in debug mode
        wilcoxon_result = wilcoxon(accuracy1, accuracy2)
        stat = wilcoxon_result.statistic
        p = wilcoxon_result.pvalue
        print("Wilcoxon result:")
        print("1. Statistics: " + wilcoxon_result.statistic.__str__())
        print("2. p-value: " + wilcoxon_result.pvalue.__str__())

        print(f"Wilcoxon signed-rank test between {name1} and {name2}: stat={stat:.4f}, p={p:.6f}")
        if p < 0.05:
            print(" -> The difference is statistically significant (p < 0.05).")
        else:
            print(" -> No significant difference (p >= 0.05).")
    except Exception as e:
        print("Unexpected error - could not perform Wilcoxon test:", e)


    # ROC plotting for top 2 models - average AUCs
    def plot_average_roc(model_name, results_entry):
        plt.figure(figsize=(8, 6))
        plt.title(f"Model {model_name} — Mean AUC: {results_entry['mean_auc']:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True)
        plt.plot([0,1],[0,1],'k--', alpha=0.6)
        plt.show()

    plot_average_roc(name1, res1)
    plot_average_roc(name2, res2)

print("Execution completed")
print("And they have lived happily ever after :))")
