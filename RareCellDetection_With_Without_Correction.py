from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sympy import symbols, Eq, solve
from sklearn.svm import SVC
import numpy as np
import random

# Set a fixed random seed
seed = 42  # You can choose any fixed value as the seed
np.random.seed(seed)
random.seed(seed)


def solve_quadratic_equations(a, b, m, n):
    """
    Where m is the number of samples predicted as HELA, and n is the number of samples predicted as Jurkat.
    """
    # Define symbolic variables
    x, y = symbols('x y')

    # Define the system of equations
    equation1 = Eq(x + y * a * 0.01 - x * b * 0.01, m)
    equation2 = Eq(y + x * b * 0.01 - y * a * 0.01, n)

    # Solve the system of equations
    solutions = solve((equation1, equation2), (x, y))
    x = solutions[x]
    y = solutions[y]
    return x, y


def savedata(root, name, filtereddata):
    filtereddata.to_excel(root + name, index=False)


if __name__ == '__main__':
    # Import data and process data
    jurkat_root = r'F:\LW_data\文章数据和程序\Fig6\Before_and_after_correction\data\Jurkat-fig3.xlsx'
    hela_root = r'F:\LW_data\文章数据和程序\Fig6\Before_and_after_correction\data\Hela-fig3.xlsx'
    jurkat_data = pd.read_excel(jurkat_root, header=None, sheet_name="cell").values
    HELA_data = pd.read_excel(hela_root, header=None, sheet_name="cell").values

    MIX1_result = []
    MIX5_result = []
    MIX10_result = []
    MIX1_result_before = []
    MIX5_result_before = []
    MIX10_result_before = []
    num = 50

    array1 = np.repeat(0, repeats=jurkat_data.shape[0])
    array2 = np.repeat(1, repeats=HELA_data.shape[0])

    X1 = np.concatenate((jurkat_data, HELA_data), axis=0)
    y1 = np.concatenate((array1, array2))
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    count = jurkat_data.shape[0]
    jurkat_data = X1[:count, :]
    HELA_data = X1[count:, :]

    for j in range(num):
        print(f"Current progress: {j + 1}/{num}")
        #### Select training and test samples of Jurkat (randomly)
        n_jurkat_samples = jurkat_data.shape[0]
        # Select the number of samples to copy, which is 50% of the total number of samples
        num_samples_to_copy = int(0.5 * n_jurkat_samples)
        # Use np.random.choice to randomly select num_samples_to_copy indices from 0 to n_jurkat_samples-1
        random_indices = np.random.choice(n_jurkat_samples, num_samples_to_copy, replace=False)
        # Get the remaining indices
        remaining_indices = np.setdiff1d(np.arange(n_jurkat_samples), random_indices)
        ### Samples of Jurkat training set and test set
        jurkat_train = jurkat_data[random_indices]
        jurkat_test = jurkat_data[remaining_indices]

        ### Select training and test samples of HELA (randomly)
        n_HELA_samples = HELA_data.shape[0]
        # Use np.random.choice to randomly select num_samples_to_copy indices from 0 to n_HELA_samples-1
        random_indices = np.random.choice(n_HELA_samples, num_samples_to_copy, replace=False)
        # Get the remaining indices
        remaining_indices = np.setdiff1d(np.arange(n_HELA_samples), random_indices)
        ### Samples of HELA training set and test set
        HELA_train = HELA_data[random_indices]
        HELA_test = HELA_data[remaining_indices]

        # Data integration
        array1 = np.repeat(0, repeats=num_samples_to_copy)
        array2 = np.repeat(1, repeats=num_samples_to_copy)
        # X is the training samples, y is the training labels
        X = np.concatenate((jurkat_train, HELA_train), axis=0)
        y = np.concatenate((array1, array2))

        ## remaining samples--HELA
        n_reHELA_samples = HELA_test.shape[0]
        # Select the number of samples to copy, which are 1%, 5%, and 10% of the number of Jurkat samples
        num_samples_to_copy_1 = int(1 / 99 * num_samples_to_copy)
        num_samples_to_copy_5 = int(5 / 95 * num_samples_to_copy)
        num_samples_to_copy_10 = int(1 / 9 * num_samples_to_copy)
        # Use np.random.choice to randomly select num_samples_to_copy indices from 0 to n_reHELA_samples-1
        random_indices_1 = np.random.choice(n_reHELA_samples, num_samples_to_copy_1, replace=False)
        random_indices_5 = np.random.choice(n_reHELA_samples, num_samples_to_copy_5, replace=False)
        random_indices_10 = np.random.choice(n_reHELA_samples, num_samples_to_copy_10, replace=False)

        HELA_1 = HELA_test[random_indices_1]
        HELA_5 = HELA_test[random_indices_5]
        HELA_10 = HELA_test[random_indices_10]

        MIX1 = np.concatenate((jurkat_test, HELA_1), axis=0)
        MIX5 = np.concatenate((jurkat_test, HELA_5), axis=0)
        MIX10 = np.concatenate((jurkat_test, HELA_10), axis=0)
        # End of finding the best parameters, output the best parameters
        # Find the best values of a and b
        ### Use the best model to find the best values of a and b
        # Used to save the percentage of the element in the second row and first column of the confusion matrix for each model
        results_percent_a = []
        results_percent_b = []
        # Perform one thousand random training and testing
        num_trials = 200
        for i in range(num_trials):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            #########import and process data over
            # Define the parameter grid
            param_grid = {
                'svc__C': [0.1, 1, 10, 100],
                'svc__gamma': [0.001, 0.01, 0.1, 1],
                'svc__kernel': ['linear', 'rbf']
            }

            # Use grid search cross-validation
            # Create a pipeline and set up GridSearchCV
            pipeline = make_pipeline(SVC())
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
            # Fit the model
            grid_search.fit(X_train, y_train)
            print("Best parameters found: ", grid_search.best_params_)
            # Extract the best parameters
            best_params = grid_search.best_params_

            # Extract the values of C, gamma, and kernel from best_params
            best_C = best_params['svc__C']
            best_gamma = best_params['svc__gamma']
            best_kernel = best_params['svc__kernel']

            clf = SVC(class_weight='balanced', C=best_C, gamma=best_gamma, kernel=best_kernel)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            # Convert the numbers in the confusion matrix to percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            results_percent_a.append(cm_percent[0, 1] * 100)  # Percentage of the element in the first row and second column of the confusion matrix
            results_percent_b.append(cm_percent[1, 0] * 100)  # Percentage of the element in the second row and first column of the confusion matrix
            # Print the current progress
            print(f"Progress: {i + 1} / {num_trials}")

        if np.var(results_percent_a) > 0:
            # Plot the density plot of a, select the best value of a according to the density plot
            plt.figure(figsize=(10, 6))
            sns.kdeplot(results_percent_a, fill=True, color='blue', warn_singular=False)
            plt.xlabel('Value (%)')
            plt.ylabel('Density')
            plt.title('Density Plot of second row, first column of confusion matrix (as percentage)')
            plt.grid(True)
            # Save the figure
            plt.savefig(r'F:\LW_data\20250124-FIG5-HELA\density_plot_a.png')
            plt.close()  # Close the figure window
            # Get the density curve data
            density_line = sns.kdeplot(results_percent_a).get_lines()[0]
            x_data, y_data = density_line.get_data()
            # Find the peak of the density estimation
            peak_index = y_data.argmax()
            peak_value = x_data[peak_index]
            a = peak_value
        else:
            a = np.median(results_percent_a)

        print(f"The value of a is: {a}")

        # Plot the density plot of b, select the best value of b according to the density plot
        if np.var(results_percent_b) > 0:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(results_percent_b, fill=True, color='blue', warn_singular=False)
            plt.xlabel('Value (%)')
            plt.ylabel('Density')
            plt.title('Density Plot of second row, first column of confusion matrix (as percentage)')
            plt.grid(True)
            # Save the figure
            plt.savefig(r'F:\LW_data\20250124-FIG5-HELA\density_plot_b.png')
            plt.close()  # Close the figure window
            # Get the density curve data
            density_line = sns.kdeplot(results_percent_b).get_lines()[0]
            x_data, y_data = density_line.get_data()
            # Find the peak of the density estimation
            peak_index = y_data.argmax()
            peak_value = x_data[peak_index]
            b = peak_value
        else:
            b = np.median(results_percent_b)
        a = float(a)
        b = float(b)
        # End of finding the best values of a and b

        # Define the parameter grid
        param_grid = {
            'svc__C': [0.1, 1, 10, 100],
            'svc__gamma': [0.001, 0.01, 0.1, 1],
            'svc__kernel': ['linear', 'rbf']
        }

        # Use grid search cross-validation
        # Create a pipeline and set up GridSearchCV
        pipeline = make_pipeline(SVC())
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
        # Fit the model
        grid_search.fit(X, y)
        print("Best parameters found: ", grid_search.best_params_)
        # Extract the best parameters
        best_params = grid_search.best_params_

        # Extract the values of C, gamma, and kernel from best_params
        best_C = best_params['svc__C']
        best_gamma = best_params['svc__gamma']
        best_kernel = best_params['svc__kernel']

        #### Output the proportions of MIX1, MIX5, and MIX10 samples
        ### Fit the best model to the training set for MIX1
        best_model = SVC(class_weight='balanced', C=best_C, gamma=best_gamma, kernel=best_kernel)
        best_model.fit(X, y)
        y_pred = best_model.predict(MIX1)
        ## Where m is the number of samples predicted as HELA, and n is the number of samples predicted as Jurkat
        m = sum(y_pred)
        count = len(y_pred)
        n = count - m
        print(a, b, m, n)
        percentage = 100 * m / count
        MIX1_result_before.append(percentage)
        # Call the function and get the solutions
        HELA_number, jurkat_number = solve_quadratic_equations(a, b, m, n)
        percentage = 100 * HELA_number / (HELA_number + jurkat_number)
        print("###########################################################################")
        print(f"The predicted percentage of HELA in MIX1 samples is: {percentage}%")
        print("###########################################################################")
        MIX1_result.append(percentage)
        ### Fit the best model to the training set for MIX5
        y_pred = best_model.predict(MIX5)
        ## Where m is the number of samples predicted as HELA, and n is the number of samples predicted as Jurkat
        m = sum(y_pred)
        count = len(y_pred)
        n = count - m
        print(a, b, m, n)
        percentage = 100 * m / count
        MIX5_result_before.append(percentage)
        # Call the function and get the solutions
        HELA_number, jurkat_number = solve_quadratic_equations(a, b, m, n)
        percentage = 100 * HELA_number / (HELA_number + jurkat_number)
        print("###########################################################################")
        print(f"The predicted percentage of HELA in MIX5 samples is: {percentage}%")
        print("###########################################################################")
        MIX5_result.append(percentage)
        ### Fit the best model to the training set for MIX10
        y_pred = best_model.predict(MIX10)
        ## Where m is the number of samples predicted as HELA, and n is the number of samples predicted as Jurkat
        m = sum(y_pred)
        count = len(y_pred)
        n = count - m
        print(a, b, m, n)
        percentage = 100 * m / count
        MIX10_result_before.append(percentage)
        # Call the function and get the solutions
        HELA_number, jurkat_number = solve_quadratic_equations(a, b, m, n)
        percentage = 100 * HELA_number / (HELA_number + jurkat_number)
        print("###########################################################################")
        print(f"The predicted percentage of HELA in MIX10 samples is: {percentage}%")
        print("###########################################################################")
        MIX10_result.append(percentage)
    # Specify the output file path and file name, modify according to your actual needs
    output_path = r'F:\LW_data\文章数据和程序\Fig6\Before_and_after_correction'
    output_file = pd.DataFrame({"MIX1": np.array(MIX1_result),
                                "MIX5": np.array(MIX5_result),
                                "MIX10": np.array(MIX10_result),
                                "MIX1_b": np.array(MIX1_result_before),
                                "MIX5_b": np.array(MIX5_result_before),
                                "MIX10_b": np.array(MIX10_result_before)
                                })

    savedata(output_path, r"\20241107statistics_of_MIX_sample_2.xlsx", output_file)