import pandas as pd
from sklearn import linear_model
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import numpy
from sklearn.model_selection import LeaveOneOut


df_timbre = pd.read_pickle('./pkl/song_data_timbre_pitch.pkl')
start_time = time.time()
print('Number of songs:', len(df_timbre))


# Create X and y
X = []
y = []
for i in range(df_timbre.shape[0]):
    x = []
    y.append(df_timbre.iloc[i][-1])
    for j in range(len(df_timbre.iloc[i][:-1])):
        x.append(df_timbre.iloc[i][:-1][j])
    X.append(x)
with open('./pkl/X.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('./pkl/y.pkl', 'wb') as f:
    pickle.dump(y, f)

# This can be used instead of the above section to load already created X.pkl and y.pkl
# with open('./pkl/X.pkl', 'rb') as f:
#     X = pickle.load(f)
# with open('./pkl/y.pkl', 'rb') as f:
#     y = pickle.load(f)
# print('Done loading X, y')


# # DELETE
# yy = []
# loo = LeaveOneOut()
# for train_index, test_index in loo.split(X):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     # X_train = X[train_index]
#     if test_index[0] % 500 == 0:
#         print(test_index)
#     X_train = [X[i] for i in train_index]
#     X_test = X[test_index[0]]
#     y_train = [y[i] for i in train_index]
#     y_test = y[test_index[0]]
#     reg = linear_model.LinearRegression()
#     reg.fit(X_train, y_train)
#     y_pred = reg.predict([X_test])[0]
#     yy.append([y_pred, y_test])
#
# yandy = list(map(list, zip(*yy)))
# print('mse', sklearn.metrics.mean_squared_error(yandy[0], yandy[1]), '\n')



t2 = time.time()
n_alphas = 100
alpha_ridge = np.logspace(1e-1, 4, n_alphas)
# DELETE THIS?!
# alpha_ridge = np.logspace(1e-10, 1e2, n_alphas)
reg = linear_model.RidgeCV(alphas=alpha_ridge)
reg.fit(X, y)
best_alpha = reg.alpha_
preds = reg.predict(X)

print('MSE:', sklearn.metrics.mean_squared_error(y, preds))
print('mae:', sklearn.metrics.mean_absolute_error(y, preds))

error = [preds[i] - y[i] for i in range(len(preds))]
print('errors:', error)
# print(min(error), max(error))
# [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 20, 30, 40]
plt.figure(1)
plt.hist(error, normed=True, bins=104)
plt.ylabel('Probability')
plt.xlabel(r'$\hat y - \bar{y}$')
# plt.show()
plt.savefig('year_prediction_histogram.png', bbox_inches='tight')
plt.close(1)

count1 = 0
count3 = 0
count5 = 0
count10 = 0
countinf = 0

for i in range(len(y)):
    diff = abs(y[i] - preds[i])
    if diff < 1:
        count1 += 1
    elif diff < 3:
        count3 += 1
    elif diff < 5:
        count5 += 1
    elif diff < 10:
        count10 += 1
    else:
        countinf += 1

print('Prediction accuracy from actual')
print('<1:', count1/len(y)*100)
print('<3:', count3/len(y)*100)
print('<5:', count5/len(y)*100)
print('<10:', count10/len(y)*100)
print('>10:', countinf/len(y)*100)

# print("--- %s seconds ---" % (time.time() - t2))

# Plot of coefficients as a function of regularization
n_alphas = 200
# DELETE
# alphas = np.logspace(0, 8, n_alphas)
alphas = np.logspace(-2, 8, n_alphas)
coefs = []
for a in alphas:
    reg = linear_model.RidgeCV(alphas=[a], store_cv_values=True)
    reg.fit(X, y)
    coefs.append(list(reg.coef_))
    # numpy.apply_along_axis(numpy.mean(a))
    # print(numpy.mean(reg.cv_values_, axis=0))

plt.figure(2)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.axvline(x=best_alpha, color='gray')
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('lambda')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
# plt.show()
plt.savefig('coefficients_vs_regularization.png', bbox_inches='tight')
plt.close(2)
print("--- %s seconds ---" % (time.time() - start_time))
