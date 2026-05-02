install:
	pip install -r requirements.txt

prep:
	python src/data_prep.py

explore:
	python src/data_exploration.py

unsupervised:
	python src/unsupervised_analysis.py

tree:
	python src/decision_tree.py

baseline_model: tree

svm:
	python src/svm_rbf.py

logreg:
	python src/logreg.py

pca_logreg:
	python src/pca_logreg.py

test:
	pytest tests/