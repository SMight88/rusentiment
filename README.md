RuSentiment
=======================================================

Создать каталог data/embeddings, скачать и распаковать в данный каталог
VK embeddings отсюда: http://text-machine.cs.uml.edu/projects/rusentiment/

Для запуска ноутбуков потребуется исходный датасет RuSentiment.
Создаем каталог data/dataset и складываем туда файлы из https://github.com/text-machine-lab/rusentiment/tree/master/Dataset

В каталоге notebooks содержатся Jupyter notebooks с экспериментами:
- baseline.ipynb - эксперименты с бейзлайном из https://github.com/text-machine-lab/rusentiment/
- train_araneum.ipynb - использован предобученный fasttext от Rusvectores - http://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz
- train.ipynb - итоговые эксперименты
- workflow.ipynb - наброски workflow + тестирование

Попробованы несколько моделей. Результаты на тестовой выборке:
- LogisticRegression: F1 0.694
- LinearSVC: F1 0.696
- GradientBoostingClassifier: F1 0.690
- SVC: F1 0.689
- MLP (final model): F1 0.738

Результаты из original paper для лучшей модели: F1 0.728.

Планы:
- Использовать fasttext VK word embeddings. [Эмбеддинги из оригинальной статьи](http://text-machine.cs.uml.edu/projects/rusentiment/) предоставлены только в формате .vec, что дает возможность загрузить их только, как w2v без информации о символьных н-граммах.
- Тюнинг моделей ...

Original paper: http://aclweb.org/anthology/C18-1064