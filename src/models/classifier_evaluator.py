class ClassifierEvaluator:
    def __init(self):
        pass

    @staticmethod
    def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return clf.score(X_test, y_test)

    @staticmethod
    def evaluate_classifiers(classifiers, X_train, y_train, X_test, y_test):
        results = {}
        for name, clf in classifiers.items():
            score = ClassifierEvaluator.evaluate_classifier(clf, X_train, y_train, X_test, y_test)
            results[name] = score
        return results
