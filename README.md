# regression
This repository preserves an implementation of Linear and Logistic Regression, using a Stochastic and Batch Gradient Descent

# Linear Regression
    x = np.array([1, 2, 4, 3, 5])  # x variable
    y = np.array([1, 3, 3, 2, 5])  # y variable


    # Creates a linear_regression object
    lm_stochastic = regression.linear_regression()
    
    # fit a model using stochastic gradient descent algorithm
    lm_stochastic.fit(y, x, method='stochastic', epochs=10000, epsilon= 0.000000001, plot_it=True, learning_rate= 0.001)
    
    # plot the sse plot over each epoch
    lm_stochastic.generate_sse_plot
    
    # prediction
    yhat_stochastic = lm_stochastic.predict(x)
    
    print(yhat_stochastic)
    print(f'\nStochastic Gradient Descent Info\nR square: {lm_stochastic.score(y,x)[0]}\nAdj R square: {lm_stochastic.score(y,x)                        [1]}\nCoeffs: {lm_stochastic.coefs}\tP-values: {lm_stochastic.p_values}\nIntercept: {lm_stochastic.intercept}\nRMSE:                    {lm_stochastic.get_RMSE(y,yhat_stochastic)}\n')

    lm_batch = regression.linear_regression()
    lm_batch.fit(y,x, method = 'batch', epochs= 10000,epsilon= 0.00000001, plot_it=True, learning_rate= 0.01)
    
    # plot the sse plot over each epoch
    lm_batch.generate_sse_plot
    yhat_batch = lm_batch.predict(x)
    
    print(yhat_batch)
    print(f'\nBatch Gradient Descent Info\nR square: {lm_batch.score(y,x)[0]}\nAdj R square: {lm_batch.score(y,x)[1]}\nCoeffs:                  {lm_batch.coefs}\tP-values: {lm_batch.p_values}\nIntercept: {lm_batch.intercept}\nRMSE: {lm_batch.get_RMSE(y,yhat_batch)}\n')










    x = [2.7810836,1.465489372,3.396561688,1.38807019,3.06407232,7.627531214,5.332441248,6.922596716,8.675418651,7.673756466]
    y = [0,0,0,0,0,1,1,1,1,1]

    logistic = regression.logistic_model()
    
    # train the linear model
    logistic.fit(y,x, epochs=100, accuracy_threshold=1, learning_rate=0.3)
    
    print(f'Coefficients : {logistic.coefs}\tIntercept : {logistic.intercept}')
    # Generate cost plot
    logistic.generate_accuracy_plot
    
    # Prediction
    yhat = logistic.predict(x)
    
    print(f'Classifiers : {logistic.classifier(yhat,0.5)}\nAccuracy: {logistic.accuracy(y,yhat)} %')
