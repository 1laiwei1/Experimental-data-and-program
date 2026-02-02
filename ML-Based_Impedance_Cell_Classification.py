from sklearn.metrics import make_scorer, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib.colors import to_hex
import matplotlib.colors as mcolors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

# Import data and process data
jurkat_root = r'F:\LW_data\文章数据和程序\Fig3-5\(1,2,2,5)features_Test_prediction\data\20250220-J-HELA\Jurkat-fig3.xlsx'
hela_root = r'F:\LW_data\文章数据和程序\Fig3-5\(1,2,2,5)features_Test_prediction\data\20250220-J-HELA\Hela-fig3.xlsx'

jurkat_data = pd.read_excel(jurkat_root, header=None, sheet_name="cell").values
hela_data = pd.read_excel(hela_root, header=None, sheet_name="cell").values
count = jurkat_data.shape[0]

array1 = np.repeat(0, repeats=jurkat_data.shape[0])
array2 = np.repeat(1, repeats=hela_data.shape[0])

X1 = np.concatenate((jurkat_data, hela_data), axis=0)
y1 = np.concatenate((array1, array2))

#Save the original raw data for subsequent display
diameter_raw_j = X1[:count, 0]
diameter_raw_T = X1[count:, 0]

mid_raw_j = X1[:count, 1:3]
mid_raw_T = X1[count:, 1:3]

high_raw_j = X1[:count, 3:5]
high_raw_T = X1[count:, 3:5]

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)

jurkat_data = X1[:count, :]
hela_data = X1[count:, :]

# Selection of the number of Jurkat and HELA in the training dataset
n_jurkat = jurkat_data.shape[0]
n_hela = hela_data.shape[0]
if n_jurkat < n_hela:
    com_number = n_jurkat
else:
    com_number = n_hela

# Set the random seed
np.random.seed(45)

# Use np.random.choice to randomly select num_samples_to_copy indices from 0 to n_jurkat-1
jurkat_random_indices = np.random.choice(n_jurkat, com_number, replace=False)
# Use np.random.choice to randomly select num_samples_to_copy indices from 0 to n_hela-1
hela_random_indices = np.random.choice(n_hela, com_number, replace=False)
jurkat_data = jurkat_data[jurkat_random_indices]
hela_data = hela_data[hela_random_indices]
diameter_raw_j = diameter_raw_j[jurkat_random_indices]
diameter_raw_T = diameter_raw_T[hela_random_indices]

mid_raw_j = mid_raw_j[jurkat_random_indices]
mid_raw_T = mid_raw_T[hela_random_indices]

high_raw_j = high_raw_j[jurkat_random_indices]
high_raw_T = high_raw_T[hela_random_indices]

# Data integration
array1 = np.repeat('Jurkat', repeats=com_number)
array2 = np.repeat('HELA', repeats=com_number)

array3 = np.repeat(0, repeats=com_number)
array4 = np.repeat(1, repeats=com_number)
diameter_raw = np.concatenate((diameter_raw_j, diameter_raw_T), axis=0)
mid_raw = np.concatenate((mid_raw_j, mid_raw_T), axis=0)
high_raw = np.concatenate((high_raw_j, high_raw_T), axis=0)

print(diameter_raw)
features_array = np.concatenate((jurkat_data, hela_data), axis=0)

labels = np.concatenate((array1, array2))
c_label = np.concatenate((array3, array4))

# Generate indices for the training set and test set
# Calculate the total number of samples
num_samples = com_number * 2

# Split the training set and test set in a 4:1 ratio
# Generate an index array
indices = np.random.permutation(num_samples)

# Calculate the number of samples in the training set and test set
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size

# Extract the indices of the training set and test set
train_indices = indices[:train_size]
test_indices = indices[train_size:]
train_label = labels[train_indices]
test_label = labels[test_indices]

# Define the LDA model
lda1 = LDA(n_components=1)
lda2 = LDA(n_components=1)

# Apply LDA dimensionality reduction to the specified column(s)
X_lda1 = lda1.fit_transform(features_array[:, 1:3], labels)
X_lda2 = lda2.fit_transform(features_array[:, 3:5], labels)
print("Jurkat:")
print(pd.DataFrame(X_lda1).head(20))
print(labels[0:20])
print("\n")
print("HELA:")
print(pd.DataFrame(X_lda1).tail(20))

print(pd.DataFrame(X_lda2).head(6))
print(pd.DataFrame(X_lda2).tail(6))
# Combine the dimensionality reduction results into a new DataFrame
reduced_data = pd.DataFrame(np.hstack((X_lda1, X_lda2)), columns=['10MHZ', '28MHZ'])

# Select color values for the color scheme
# Customize RGB colors
color_1_rgb = (44, 177, 136)  # Jurkat
color_5_rgb = (160, 0, 200)  # HELA

# Convert RGB to hexadecimal
color_1_hex = to_hex([c / 255.0 for c in color_1_rgb])
color_5_hex = to_hex([c / 255.0 for c in color_5_rgb])


JHE_c = {
    'Jurkat': color_1_hex,
    'HELA': color_5_hex
}

JHE_cmap = mcolors.ListedColormap([
    (44 / 255, 177 / 255, 136 / 255),
    (160 / 255, 0 / 255, 200 / 255),
])


# Plot a confusion matrix for the first column, diameter
diameter_train = features_array[train_indices, 0].reshape(-1, 1)
diameter_test = features_array[test_indices, 0].reshape(-1, 1)
c_test_label = c_label[test_indices]

# Define the parameter grid
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [0.001, 0.01, 0.1, 1],
    'svc__kernel': ['linear', 'rbf']
}
# Create a pipeline and set up GridSearchCV
pipeline = make_pipeline(SVC())
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
# Fit the model
grid_search.fit(diameter_train, train_label)
print("Best parameters found: ", grid_search.best_params_)
# Extract the best parameters
best_params = grid_search.best_params_

# Extract the values of C, gamma, and kernel from best_params
best_C = best_params['svc__C']
best_gamma = best_params['svc__gamma']
best_kernel = best_params['svc__kernel']
# Perform SVM classification on the first column, diameter
clf = SVC(class_weight='balanced', C=best_C, gamma=best_gamma, kernel=best_kernel)
clf.fit(diameter_train, train_label)
y_pred = clf.predict(diameter_test)
cm1 = confusion_matrix(test_label, y_pred)

# Plot a confusion matrix for the 2nd and 3rd columns, 10MHz
train_10Mhz = features_array[train_indices, 1:3]
test_10Mhz = features_array[test_indices, 1:3]

# Define the parameter grid
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [0.001, 0.01, 0.1, 1],
    'svc__kernel': ['linear', 'rbf']
}

# Create a pipeline and set up GridSearchCV
pipeline = make_pipeline(SVC())
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
# Fit the model
grid_search.fit(train_10Mhz, train_label)
print("Best parameters found: ", grid_search.best_params_)
# Extract the best parameters
best_params = grid_search.best_params_
best_C = best_params['svc__C']
best_gamma = best_params['svc__gamma']
best_kernel = best_params['svc__kernel']
# Perform SVM classification on the 2nd and 3rd columns, 10MHz
clf = SVC(class_weight='balanced', C=best_C, gamma=best_gamma, kernel=best_kernel)
clf.fit(train_10Mhz, train_label)
y_pred = clf.predict(test_10Mhz)
cm2 = confusion_matrix(test_label, y_pred)

# Plot a confusion matrix for the 4th and 5th columns, 28MHz
train_28Mhz = features_array[train_indices, 3:5]
test_28Mhz = features_array[test_indices, 3:5]

# Define the parameter grid
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [0.001, 0.01, 0.1, 1],
    'svc__kernel': ['linear', 'rbf']
}
# Create a pipeline and set up GridSearchCV
pipeline = make_pipeline(SVC())
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
# Fit the model
grid_search.fit(train_28Mhz, train_label)
print("Best parameters found: ", grid_search.best_params_)
# Extract the best parameters
best_params = grid_search.best_params_
best_C = best_params['svc__C']
best_gamma = best_params['svc__gamma']
best_kernel = best_params['svc__kernel']
# Perform SVM classification on the 4th and 5th columns, 28MHz
clf = SVC(class_weight='balanced', C=best_C, gamma=best_gamma, kernel=best_kernel)
clf.fit(train_28Mhz, train_label)
y_pred = clf.predict(test_28Mhz)
cm3 = confusion_matrix(test_label, y_pred)

# Plot a confusion matrix for columns 1 to 5
# Perform SVM classification on columns 1 to 5
# Define the parameter grid
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [0.001, 0.01, 0.1, 1],
    'svc__kernel': ['linear', 'rbf']
}
# Create a pipeline and set up GridSearchCV
pipeline = make_pipeline(SVC())
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
# Fit the model
grid_search.fit(features_array[train_indices, :], train_label)
print("Best parameters found: ", grid_search.best_params_)
# Extract the best parameters
best_params = grid_search.best_params_
best_C = best_params['svc__C']
best_gamma = best_params['svc__gamma']
best_kernel = best_params['svc__kernel']
clf = SVC(class_weight='balanced', C=best_C, gamma=best_gamma, kernel=best_kernel)
clf.fit(features_array[train_indices, :], train_label)
y_pred = clf.predict(features_array[test_indices, :])
cm4 = confusion_matrix(test_label, y_pred)

# Create a DataFrame to save the original diameter data and labels
df_diameter_raw = pd.DataFrame({
    'diameter_raw': diameter_raw[test_indices],
    'labels': test_label
})

# 10MHz
df_10M = pd.DataFrame({
    'Opacity_X': features_array[test_indices, 1],
    'Opacity_Y': features_array[test_indices, 2],
    'labels': test_label
})

# 28MHz
df_28M = pd.DataFrame({
    'Opacity_X': features_array[test_indices, 3],
    'Opacity_Y': features_array[test_indices, 4],
    'labels': test_label
})

# Create a 2x4 layout plot
fig = plt.figure(figsize=(20, 10))  # Increase the height of the figure
fig.subplots_adjust(wspace=0.4, hspace=0.6)  # Increase the horizontal and vertical spacing between subplots
# Create subplots
axs = [fig.add_subplot(2, 4, i + 1) for i in range(3)]
ax3d = fig.add_subplot(2, 4, 4, projection='3d')  # Place the 3D plot in the fourth position of the first row
# Create the positions for the subsequent four subplots
heatmap_axes = [fig.add_subplot(2, 4, i + 5) for i in range(4)]

# Ensure the 'labels' column is ordered
df_diameter_raw['labels'] = pd.Categorical(df_diameter_raw['labels'], categories=['Jurkat', 'HELA'], ordered=True)

# Plot the violin plot
sns.violinplot(x='labels', y='diameter_raw',data=df_diameter_raw,split=False,inner='box',scale='width',
               hue_order=['Jurkat', 'HELA'], palette=JHE_c,width=0.5, ax=axs[0], flierprops=dict(markersize=2) )
if axs[0].get_legend() is not None:
    axs[0].get_legend().remove()
axs[0].set_title('Low frequency', pad=20)
axs[0].set_xlabel('Category')
axs[0].set_ylabel('Diameter(um)')

# Set the y-axis to integer format
axs[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))




# Plot the 10MHz scatter plot
sns.scatterplot(x='Opacity_X', y='Opacity_Y', hue='labels', data=df_10M, hue_order=['HELA', 'Jurkat'],
                palette=JHE_c, s=10, edgecolor=None, alpha=0.5, linewidth=0, ax=axs[1])
axs[1].legend(title='Classes', loc='lower right', fontsize='x-small', title_fontsize='small', handletextpad=0.4,
              borderpad=0.4, handlelength=1.5)
axs[1].set_title('Medium frequency', pad=20)
axs[1].set_xlabel('Opacity_X(Standardized)')
axs[1].set_ylabel('Opacity_Y(Standardized)')
axs[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
axs[1].yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))

# Plot the 28MHz scatter plot
sns.scatterplot(x='Opacity_X', y='Opacity_Y', hue='labels', data=df_28M, hue_order=['HELA', 'Jurkat'],
                palette=JHE_c, s=10, edgecolor=None, alpha=0.5, linewidth=0, ax=axs[2])
axs[2].legend(title='Classes', loc='lower right', fontsize='x-small', title_fontsize='small', handletextpad=0.4,
              borderpad=0.4, handlelength=1.5)
axs[2].set_title('High frequency', pad=20)
axs[2].set_xlabel('Opacity_X(Standardized)')
axs[2].set_ylabel('Opacity_Y(Standardized)')
axs[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
axs[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))

reduced_data_test = reduced_data.iloc[test_indices]
print(reduced_data_test.head(20))  # Display the first 5 rows

# Plot the 3D scatter plot
# Create a mapping dictionary
label_mapping = {0: 'Jurkat', 1: 'HELA'}

# Create a new mapping for c_test_label
mapped_labels = [label_mapping[label] for label in c_test_label]
ax3d.set_xlabel('10MHZ')
ax3d.set_ylabel('28MHZ')
ax3d.set_zlabel('Diameter')
ax3d.set_title('3D scatter plot', pad=20)
ax3d.xaxis.set_visible(False)
ax3d.yaxis.set_visible(False)

scatter = ax3d.scatter(reduced_data_test['10MHZ'], reduced_data_test['28MHZ'], df_diameter_raw['diameter_raw'],
                       c=c_test_label, cmap=JHE_cmap, s=3, alpha=0.5)
legend = ax3d.legend(*scatter.legend_elements(), loc='upper left', title='Classes', fontsize='x-small',
                     title_fontsize='small', handletextpad=0.4, borderpad=0.4, handlelength=1.5)
for text, label in zip(legend.texts, ['Jurkat', 'HELA']):
    text.set_text(label)
ax3d.add_artist(legend)

# Set the X, Y, and Z axes of the 3D plot to integer ticks
ax3d.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax3d.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax3d.zaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Plot the confusion matrix heatmap
def plot_confusion_matrix(ax, cm, title):
    # Convert the confusion matrix to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy = np.trace(cm) / np.sum(cm)

    # Create annotations
    annotations = np.array([[f'{percent:.2%}\n\n{value}'
                             for value, percent in zip(row, row_percent)]
                            for row, row_percent in zip(cm, cm_percent)])

    # Define custom class labels
    class_labels = ['Jurkat', 'HELA']

    # Create a heatmap with custom labels
    sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues', cbar=False, square=True, ax=ax,
                annot_kws={"size": 10}, linewidths=0.5,
                xticklabels=class_labels, yticklabels=class_labels)

    # Set labels and title
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title(f'{title}\nTest accuracy= {accuracy:.2%}', pad=20)

plot_confusion_matrix(heatmap_axes[0], cm1, '1 feature')
plot_confusion_matrix(heatmap_axes[1], cm2, '2 features')
plot_confusion_matrix(heatmap_axes[2], cm3, '2 features')
plot_confusion_matrix(heatmap_axes[3], cm4, '5 features')
plt.show()