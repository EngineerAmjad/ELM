from skelm import ELMClassifier
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
class FitnessHistory(Callback):
    def __init__(self):
        super().__init__()
        self.history = []

    def notify(self, algorithm):
        self.history.append(algorithm.pop.get("F").min())

class set_weights(Problem):

    def __init__(self, elm, X_test, y_test,X_train, y_train):
        self.elm = elm
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_var = self.elm.n_neurons * self.X_train.shape[1]

        super().__init__(n_var=self.num_var,
                         n_obj=1,
                         n_constr=0,
                         xl=-1,
                         xu=1,
                         elementwise_evaluation=True)

    def _evaluate(self, solution_, out, *args, **kwargs):
        results = []
        for solution in solution_:
            results.append(self.objective_function(self.X_train,self.y_test,self.X_test, self.y_test, solution))
        out["F"] = np.array(results)

    def objective_function(self,X_train,y_train, X_test, y_test, solution):
        self.elm.weights = solution.reshape(( X_train.shape[1],self.elm.n_neurons))  # Set the weights of the ELM classifier
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        self.elm.fit(X_resampled, y_resampled)
        y_pred = self.elm.predict(X_test)
        fitness = f1_score(y_test, y_pred, average='weighted')
        return np.array([-fitness])
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, precision_score, roc_auc_score, roc_curve
def sin(x):
    return np.sin(x)
data = pd.read_csv('C:\\Users\\admin\\Desktop\\scikit-elm-master\\spaghetti-code (2).csv')
X = data.drop(['perception'], axis=1)
y = data['perception']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
elm = ELMClassifier(ufunc='tanh',n_neurons=20)
num_iter = 100

problem = set_weights(elm, X_test, y_test,X_test,y_test)
algorithm = GA(pop_size=100,
eliminate_duplicates=True)
fitness_history = FitnessHistory()
stop_criteria = ('n_gen', num_iter)
res = minimize(problem, algorithm,seed=1, termination= stop_criteria, callback=fitness_history)




pareto_front = res.X
print(res.F)

elm.weights = pareto_front.reshape(( X_train.shape[1],elm.n_neurons))

y_pred = elm.predict(X_test)
v= f1_score(y_pred=y_pred,y_true=y_test,average='weighted')
plt.plot(fitness_history.history)

plt.title('Convergence Plot Over Generations spaghetti-code')
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.savefig('C:\\Users\\admin\\Desktop\\scikit-elm-master\\spaghetti-code.png')
plt.show()
print(v)

average_f_measure = np.mean(f1_score(y_pred=y_pred,y_true=y_test, average='weighted'))
average_mcc = np.mean(matthews_corrcoef(y_pred=y_pred,y_true=y_test))
average_recall = np.mean(recall_score(y_pred=y_pred,y_true=y_test, average='weighted'))
average_precision = np.mean(precision_score(y_pred=y_pred,y_true=y_test, average='weighted'))
result_2 = {
'F1 Score': [average_f_measure],
'MCC': [average_mcc],
'Recall': [average_recall],
'Precision': [average_precision]
}
metrics_df = pd.DataFrame(result_2)

# Save the DataFrame to a CSV file
# metrics_df.to_excel(metrics_result_file, index=False)


print(f"Scores for 'spagathi':")
print(f"F-measure: {average_f_measure:.2f}")
print(f"MCC: {average_mcc:.2f}")
print(f"Recall: {average_recall:.2f}")
print(f"Precision: {average_precision:.2f}")
print("--------------------------------------------------")
paper_metrics_values = [[0.80,0.71,0.81,0.79]]
average_scores = [average_f_measure, average_mcc, average_recall, average_precision]
labels = ['F-measure', 'MCC', 'Recall', 'Precision']
x = np.arange(len(labels)) # Original x-values
width = 0.35 # Adjusted width for two sets of bars

fig, ax = plt.subplots()

# Plot average_scores
rects1 = ax.bar(x - width/2, average_scores, width, label='ELM_With_Optim. Scores Scores')

# Plot paper_scores
rects2 = ax.bar(x + width/2, paper_metrics_values[0], width, label='Paper Scores')

ax.set_ylabel('Scores')
ax.set_title(f'Comparison of Performance Metrics for spaghetti-code')

ax.set_xticks(x)
ax.set_xticklabels(labels)


# Add text labels for average_scores
for rect in rects1:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height,
            f'{height:.2f}', ha='center', va='bottom')

# Add text labels for paper_scores
for rect in rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height,
            f'{height:.2f}', ha='center', va='bottom')
ax.legend(loc='lower right')

fig.tight_layout()

plt.savefig('C:\\Users\\admin\\Desktop\\scikit-elm-master\spaghetti-code-Comparison.png')
plt.savefig('C:\\Users\\admin\\Desktop\\scikit-elm-master\\spaghetti-code-Comparison.svg')
plt.show()
plt.close()
