import pandas as pd
from sklearn import linear_model
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle


df_timbre = pd.read_pickle('./pkl/song_data_timbre_pitch.pkl')
start_time = time.time()
print('Number of songs:', len(df_timbre))


# Use if pkl/X.pkl or pkl/y.pkl do not exist
# Create X and y
# X = []
# y = []
# for i in range(df_timbre.shape[0]):
#     x = []
#     y.append(df_timbre.iloc[i][-1])
#     for j in range(len(df_timbre.iloc[i][:-1])):
#         x.append(df_timbre.iloc[i][:-1][j])
#     X.append(x)
# with open('./pkl/X.pkl', 'wb') as f:
#     pickle.dump(X, f)
# with open('./pkl/y.pkl', 'wb') as f:
#     pickle.dump(y, f)

# Loads already created pkl/X.pkl and pkl/y.pkl
with open('./pkl/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('./pkl/y.pkl', 'rb') as f:
    y = pickle.load(f)
print('Done loading X, y')


n_alphas = 100
alpha_ridge = np.logspace(1e-1, 4, n_alphas)
reg = linear_model.RidgeCV(alphas=alpha_ridge)
reg.fit(X, y)
best_alpha = reg.alpha_
preds = reg.predict(X)

print('MSE:', sklearn.metrics.mean_squared_error(y, preds))
print('mae:', sklearn.metrics.mean_absolute_error(y, preds))

error = [preds[i] - y[i] for i in range(len(preds))]
print('errors:', error)

plt.figure(1)
plt.hist(error, normed=True, bins=104)
plt.ylabel('Probability')
plt.xlabel(r'$\hat y - \bar{y}$')
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


# Plot of coefficients as a function of regularization
n_alphas = 200
alphas = np.logspace(-2, 8, n_alphas)
coefs = []
for a in alphas:
    reg = linear_model.RidgeCV(alphas=[a], store_cv_values=True)
    reg.fit(X, y)
    coefs.append(list(reg.coef_))

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
plt.savefig('coefficients_vs_regularization.png', bbox_inches='tight')
plt.close(2)
print("--- %s seconds ---" % (time.time() - start_time))
