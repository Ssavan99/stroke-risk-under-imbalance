import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix 
import seaborn as sns
import time

def model_pipeline(model, train_test_data, suffix="", model_params=None):
    X_train, X_test, y_train, y_test = train_test_data
    if not model_params:
        model.fit(X_train, y_train)
        cr = evaluate_model(model, X_test, y_test, suffix)
        return model, cr
    
    grid = GridSearchCV(model, model_params, scoring='recall', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(model.best_params_)
    cr = evaluate_model(grid, X_test, y_test, suffix)
    return grid, cr

def evaluate_model(model, X_test, y_test, suffix=""):
    model_predictions = model.predict(X_test)
    print(type(model).__name__ + suffix)
    cr = classification_report(y_test, model_predictions)
    print(cr)
    print("++++++++++++++++++++++++++++++++++++++++++++")
    cf_matrix = confusion_matrix(y_test, model_predictions)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    heatmap = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    heatmap.get_figure().savefig( type(model).__name__ + suffix )
    time.sleep(0.005)
    heatmap.get_figure().clf()
    return cr
    
def run_all_models(models, data, suffix=""):
    trained_models = []
    reports = []
    for model in models:
        model_result, report = model_pipeline(model, data, suffix)
        trained_models.append(model_result)
        reports.append(report)
    return trained_models, reports