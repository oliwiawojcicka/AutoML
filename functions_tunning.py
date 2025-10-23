import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_score, train_test_split
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV 
import matplotlib.pyplot as plt
from numpy import maximum

def load_and_split(path, target_name, test_size=0.2, random_state=42):
    data = pd.read_csv(path)
    X=  data.drop(columns=[target_name])
    y = data[target_name]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def tune(model, param_grid, X_train, y_train, scoring='roc_auc', n_iter_random=50, n_iter_bayes=50):
    random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=n_iter_random,
    cv=3, 
    scoring=scoring,
    verbose=1,
    random_state=42,
    n_jobs=-1 
    )

    random_search.fit(X_train, y_train)    
    
    opt = BayesSearchCV(
    estimator=model,
    search_spaces=param_grid,
    n_iter=n_iter_bayes,
    cv=3,
    scoring=scoring,      
    n_jobs=-1,
    random_state=42,
    refit=True,
    verbose=0,
    )
    
    opt.fit(X_train, y_train)
    return random_search, opt

def tune_for_each_data(model, param_grid, list_X_train, list_y_train, scoring='roc_auc'):
    import pandas as pd

    results_rs = []
    results_bs = []
    scores_rs = []
    all_results_df = []

    for i in range(len(list_X_train)):
        rs, bs = tune(model, param_grid, list_X_train[i], list_y_train[i], scoring)

        results_rs.append(rs)
        results_bs.append(bs)

        rs_df = pd.DataFrame(rs.cv_results_)
        params_df = pd.DataFrame(rs_df['params'].tolist())
        params_df['mean_test_score'] = rs_df['mean_test_score']
        params_df['dataset'] = f'dataset_{i+1}'
        
        all_results_df.append(params_df)
        scores_rs.append(rs_df['mean_test_score'])

    full_results_df = pd.concat(all_results_df, ignore_index=True)

    return results_rs, results_bs, scores_rs, full_results_df

def plots_for_iterations(rs, bs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    random_iter = len(rs.cv_results_['mean_test_score'])
    bayes_iter = len(bs.cv_results_['mean_test_score'])

    rs_best = list(rs.cv_results_['mean_test_score'])
    bs_best = list(bs.cv_results_['mean_test_score'])
    rs_max_score = maximum.accumulate(rs.cv_results_["mean_test_score"])
    bs_max_score = maximum.accumulate(bs.cv_results_["mean_test_score"])

    axes[0].plot(range(1, random_iter + 1), rs_best, label='RandomizedSearchCV', marker='o')    
    axes[0].plot(range(1, random_iter + 1), rs_max_score, color="orange", linestyle="--", label="Running best")
    axes[0].set_title('RandomizedSearchCV')
    axes[0].set_xlabel('Liczba iteracji')
    axes[0].legend()
    axes[0].grid()
    
    axes[1].plot(range(1, bayes_iter + 1), bs_best, label='BayesSearchCV', marker='x')
    axes[1].plot(range(1, bayes_iter + 1), bs_max_score, color="orange", linestyle="--", label="Running best")
    axes[1].set_title('BayesSearchCV')
    axes[1].set_xlabel('Liczba iteracji')
    axes[1].legend()
    axes[1].grid()

    axes[0].set_ylabel('Średni wynik testu')
    
    plt.suptitle("Porównanie postępu wyników dla RandomSearch i BayesSearch (XGBoost)", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    
def search_for_default(full_results_df):

    # Wyodrębnienie nazw kolumn odpowiadających hiperparametrom
    hyperparameter = [col for col in full_results_df.columns if col not in ['mean_test_score', 'dataset']]

    # Uśrednienie wyników dla każdego zestawu hiperparametrów (po wszystkich zbiorach danych)
    df_grouped = full_results_df.groupby(hyperparameter)['mean_test_score'].mean().reset_index()

    for col in hyperparameter:
        if col in full_results_df.columns:
            df_grouped[col] = df_grouped[col].astype(full_results_df[col].dtype)
            
    # Wybór najlepszego zestawu hiperparametrów
    best_idx = df_grouped['mean_test_score'].idxmax()
    best_params = {}
    for param in hyperparameter:
        value = df_grouped.loc[best_idx, param]
        best_params[param] = value
    #best_params = df_grouped.loc[best_idx, hyperparameter].to_dict()
    best_score = df_grouped.loc[best_idx, 'mean_test_score']

    # Posortowana tabela wszystkich kombinacji hiperparametrów
    df_sorted = df_grouped.sort_values('mean_test_score', ascending=False)

    return best_params, best_score, df_sorted

def evaluate_on_test(model, best_params, X_train, y_train, X_test, y_test):
    best_model = model.__class__(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    return test_auc, classification_report(y_test, y_pred)


def analyze_tunability(model, list_X_train, list_y_train, list_best_params, default_params, scoring = 'roc_auc'):
    default_scores = []
    best_scores = []
    diff_scores = []
    
    for i in range(len(list_X_train)):
        X_train = list_X_train[i]
        y_train = list_y_train[i]
        
        default_model = model.__class__(**default_params, random_state=42)
        print(default_model)
        default_score = cross_val_score(default_model, X_train, y_train, cv=3, scoring=scoring).mean()
        default_scores.append(default_score)

        best_model = model.__class__(**list_best_params[i], random_state=42)

        best_score = cross_val_score(best_model, X_train, y_train, cv=3, scoring=scoring).mean()
        best_scores.append(best_score)
        
        diff_scores.append(best_score - default_score)
    return default_scores, best_scores, diff_scores   
    