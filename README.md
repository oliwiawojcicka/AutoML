## Zbiory danych jakich użyjemy
    1. https://www.kaggle.com/datasets/nezukokamaado/auto-loan-dataset
    2. https://www.kaggle.com/api/v1/datasets/download/ziya07/diabetes-clinical-dataset100k-rows
    3. https://www.kaggle.com/datasets/anthonytherrien/depression-dataset
    4. https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

## Plan:
    - wybór modeli
    - jakie parametry rozważamy do tych modeli (siatki parametrów)
    - w jaki sposób chcemy walidować wyniki? (czy cross walidacja?)

## Pytania:
    - jak duży zestaw parametrów wybrać?

## Wybrane modele:
    - XGBoost - Daria
    - RandomForest - Oliwia
    - ElasticNet - Ala


## Do zrobienia na przyszły tydzień:
    1. Siatka hiperparametrów dla naszego modelu
    2. Dla każdej ramki danych znależć najlepsze hiperparametry wykorzystując dwie metody:
        - RandomGridSearch
        - Bayes
    3. Czyli końcowo -> 6 zestawów hiperparametrów + analiza liczby iteracji dla każdej z dwóch metod (w ilu ietracjach znalazłyśmy okej hiperparametry)