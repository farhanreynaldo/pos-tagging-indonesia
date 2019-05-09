from pathlib import Path
from src.model import fit_and_dump, prepare_train_test
from src.utils import read_data
from src.model import load
from sklearn.metrics import classification_report
from sklearn_crfsuite import metrics
import pandas as pd

project_root = Path.cwd()
data_path = project_root / 'data'
model_path = project_root / 'model'

def main():
    corpus_path = data_path / 'Indonesian_Manually_Tagged_Corpus.tsv'
    tagged_sentences = read_data(corpus_path)
    models = ['baseline', 'with-prefix', 'with-suffix', 'with-allfix']
    seeds = [42, 29, 2, 6]
    model_performance = {}
    performance_df_list = []
    for model in models:
        for index, seed in enumerate(seeds):
            model_name = f'crf-{model}-fold{index+1:02d}-seed{seed:02d}.joblib'
            path = model_path / model_name
            
            if not path.exists():
                # prepare training and test data
                X_train, y_train, X_test, y_test = prepare_train_test(tagged_sentences, seed, path)
            
                # fit data
                crf = fit_and_dump(X_train, y_train, path)
        
            crf = load(path)
            X_train, y_train, X_test, y_test = prepare_train_test(tagged_sentences, seed, path)
            print(f'{model_name} exists') 

            # evaluation
            y_pred = crf.predict(X_test)
            metric_dict = metrics.flat_classification_report(y_test, y_pred, 
                                                            labels=crf.classes_, digits=3, 
                                                            output_dict=True)
            performance_df = pd.DataFrame(metric_dict).T.reset_index().assign(model_name = model_name)
            performance_df_list.append(performance_df)
    
    performance_path = model_path / 'model performance.csv'
    if not performance_path.exists():
        performance_dfs = pd.concat(performance_df_list).reset_index(drop=True).rename(columns={'index': 'tags_overall'})
        performance_dfs['model_type'] = performance_dfs['model_name'].str.split('-fold').str[0]
        performance_dfs.to_csv(model_path / 'model performance.csv', index=False)

if __name__ == '__main__':
    main()