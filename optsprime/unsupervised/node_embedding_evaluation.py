from .embedding_evaluation import EmbeddingEvaluation
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader

def get_emb_y(loader, encoder, device, dtype='numpy', is_rand_label=False):
	x, y,train_mask,test_mask,val_mask = encoder.get_embeddings(loader, device, is_rand_label)
	if dtype == 'numpy':
		return x,y,train_mask,test_mask,val_mask
	elif dtype == 'torch':
		return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),torch.from_numpy(train_mask).to(device),torch.from_numpy(test_mask).to(device),torch.from_numpy(val_mask).to(device)
	else:
		raise NotImplementedError


class NodeEmbEvaluation(EmbeddingEvaluation):
    def __init__(self, base_classifier, evaluator, task_type, num_tasks, device, params_dict=None, param_search=True,is_rand_label=False):
        self.is_rand_label = is_rand_label
        self.base_classifier = base_classifier
        self.evaluator = evaluator
        self.eval_metric = evaluator.eval_metric
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.device = device
        self.param_search = param_search
        self.params_dict = params_dict
        if self.eval_metric == 'rmse':
            self.gscv_scoring_name = 'neg_root_mean_squared_error'
        elif self.eval_metric == 'mae':
            self.gscv_scoring_name = 'neg_mean_absolute_error'
        elif self.eval_metric == 'rocauc':
            self.gscv_scoring_name = 'roc_auc'
        elif self.eval_metric == 'accuracy':
            self.gscv_scoring_name = 'accuracy'
        else:
            raise ValueError('Undefined grid search scoring for metric %s ' % self.eval_metric)

        self.classifier = None
    def embedding_evaluation(self, encoder, train_loader):
        encoder.eval()
        emb, y,train_mask,test_mask,val_mask = get_emb_y(train_loader, encoder, self.device, is_rand_label=self.is_rand_label)
        train_emb,train_y = emb[train_mask],y[train_mask]
        val_emb, val_y = emb[val_mask], y[val_mask]
        test_emb, test_y = emb[test_mask], y[test_mask]

        if 'classification' in self.task_type:

            if self.num_tasks == 1:
                train_raw, val_raw, test_raw = self.ee_binary_classification(train_emb, 
                                                                             train_y, 
                                                                             val_emb, 
                                                                             val_y, 
                                                                             test_emb,
				                                                             test_y)
            elif self.num_tasks > 1:
                train_raw, val_raw, test_raw = self.ee_multioutput_binary_classification(train_emb, train_y, val_emb, val_y,
				                                                                    test_emb, test_y)
            else:
                raise NotImplementedError
        else:
            if self.num_tasks == 1:
                train_raw, val_raw, test_raw = self.ee_regression(train_emb, train_y, val_emb, val_y, test_emb, test_y)
            else:
                raise NotImplementedError

        train_score = self.scorer(train_y, train_raw)

        val_score = self.scorer(val_y, val_raw)

        test_score = self.scorer(test_y, test_raw)

        return train_score, val_score, test_score
    
    def ee_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
        if self.param_search:
            params_dict = {'C': [0.001, 0.01,0.1,1,10,100,1000]}
            self.classifier = make_pipeline(StandardScaler(),
			                                GridSearchCV(self.base_classifier, params_dict, cv=5, scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
			                                )
        else:
            self.classifier = make_pipeline(StandardScaler(), self.base_classifier)


        self.classifier.fit(train_emb, np.squeeze(train_y))

        if self.eval_metric == 'accuracy':
            train_raw = self.classifier.predict(train_emb)
            val_raw = self.classifier.predict(val_emb)
            test_raw = self.classifier.predict(test_emb)
        else:
            train_raw = self.classifier.predict_proba(train_emb)[:, 1]
            val_raw = self.classifier.predict_proba(val_emb)[:, 1]
            test_raw = self.classifier.predict_proba(test_emb)[:, 1]
        return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)
