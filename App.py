import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to generate synthetic vehicle data
def generate_vehicle_data(features, classes, total_samples, test_size, class_settings):
    data = []
    for _ in range(total_samples):
        sample = {}
        selected_class = np.random.choice(classes)
        sample['Class'] = selected_class

        # Apply class-specific settings
        for feature in features:
            mean, std_dev = class_settings[selected_class].get(feature, (50, 10))
            sample[feature] = np.random.normal(mean, std_dev)

        data.append(sample)
    
    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['Class'], random_state=42)
    return df, train_df, test_df

# Function to train and evaluate multiple models
def compare_models(train_df, test_df, features):
    # Separate features and target
    X_train = train_df[features]
    y_train = train_df['Class']
    X_test = test_df[features]
    y_test = test_df['Class']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Combine scaled training and test sets
    X_scaled = np.vstack((X_train_scaled, X_test_scaled))
    y_scaled = np.concatenate((y_train, y_test))

    # Convert scaled data back to DataFrame for display
    scaled_df = pd.DataFrame(X_scaled, columns=features)
    scaled_df['Class'] = y_scaled

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42),
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB()
    }

    # Store results
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train_encoded)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test_encoded, y_pred)
        cm = confusion_matrix(y_test_encoded, y_pred)
        report = classification_report(y_test_encoded, y_pred, output_dict=True, target_names=label_encoder.classes_)
        results.append((name, acc, cm, report))

    return results, label_encoder.classes_, scaled_df, models

# Function for EDA
def perform_eda(df, scaled_df, features):
    generated_data, scaled_data = st.columns(2)
    # Display the dataset
    with generated_data:
        st.subheader("Generated Dataset")
        st.dataframe(df)

        # Download button for the generated dataset
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Generated Dataset (CSV)",
            data=csv_data,
            file_name='generated_dataset.csv',
            mime='text/csv'
        )

        st.markdown("<br><br>", unsafe_allow_html=True) # just adds space

    # Display the scaled dataset
    with scaled_data:
        st.subheader("Scaled Dataset")
        st.dataframe(scaled_df)

        # Download button for the scaled dataset
        csv_scaled_data = scaled_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Scaled Dataset (CSV)",
            data=csv_scaled_data,
            file_name='scaled_dataset.csv',
            mime='text/csv'
        )

        st.markdown("<br><br>", unsafe_allow_html=True)

    st.write("## Feature Visualization")
    # Scatter plot feature selection
    x_axis_feat, y_axis_feat = st.columns(2)
    with x_axis_feat:
        x_axis_feature = st.selectbox("Select X-axis feature", features, key="x_axis")
    with y_axis_feat:
        y_axis_feature = st.selectbox("Select Y-axis feature", features, key="y_axis", index=1)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Scatter plot with the selected features
    fig, ax = plt.subplots(figsize=(6, 2))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    sns.scatterplot(data=scaled_df, x=x_axis_feature, y=y_axis_feature, hue='Class', palette='husl', s=15, ax=ax)
    ax.set_xlabel(f"{x_axis_feature}", color='white', fontsize=8)
    ax.set_ylabel(f"{y_axis_feature}", color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.6, color='white')
    leg = ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='black', edgecolor='white', framealpha=1, fontsize=8)
    # Set the color of the legend title to white
    leg.get_title().set_color('white')
    for text in leg.get_texts():
        text.set_color('white')
    st.pyplot(fig)

    description = f"""
    This scatterplot visualizes the relationship between {x_axis_feature} and {y_axis_feature} for different vehicle classes (Bus, Bike, Car, Truck) in a synthetically generated dataset. Each point represents an instance of a vehicle, color-coded by its respective class.

    The purpose of this plot is to highlight the clustering of vehicle classes based on their characteristics in the simulated data. Since the dataset is generated synthetically, the exact positions of points and clusters may vary with each data generation. Nonetheless, general trends such as distinct groupings by class and potential overlaps are preserved, representing hypothetical relationships between the selected variables.
    """

    st.write(description)

# Function to plot results
def plot_results(results, class_names):
    st.write("## Model Comparison Results", wrap='center')

    # Extract model names and their corresponding accuracies
    model_names = [res[0] for res in results]
    accuracies = [res[1] for res in results]

    # Accuracy dotted line graph
    st.write("### Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(6, 2))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    sns.lineplot(x=model_names, y=accuracies, marker='o', linestyle=':', ax=ax, palette="viridis", color='white', markersize=5)
    ax.tick_params(colors='white', labelsize=5)
    ax.grid(True, linestyle='--', alpha=0.6, color='white')

    st.pyplot(fig)

    # Find the best models
    best_accuracy = np.max(accuracies)
    best_model_indices = [i for i, acc in enumerate(accuracies) if acc == best_accuracy]
    best_models = [results[i] for i in best_model_indices]
    if len(best_models) > 1:
        best_model_names = ", ".join([model[0] for model in best_models[:-1]]) + " and " + best_models[-1][0]
    else:
        best_model_names = best_models[0][0]

    # Find the worst models
    worst_accuracy = np.min(accuracies)
    worst_model_indices = [i for i, acc in enumerate(accuracies) if acc == worst_accuracy]
    worst_models = [results[i] for i in worst_model_indices]
    if len(worst_models) > 1:
        worst_model_names = ", ".join([model[0] for model in worst_models[:-1]]) + " and " + worst_models[-1][0]
    else:
        worst_model_names = worst_models[0][0]

    st.write(f"The figure above shows the accuracies of the trained models. **{best_model_names}** achieved the highest accuracy of **{best_accuracy*100:.2f}%**, while {worst_model_names} achieved the lowest accuracy of {worst_accuracy*100:.2f}%.")
            

    # Detailed classification reports with custom table styling
    st.write("### Detailed Classification Reports of Most Accurate Model/s")
    best_accuracy = np.max(accuracies)
    best_models = [res for res in results if res[1] == best_accuracy]

    # Arrange the best models in columns
    columns = st.columns(3)
    for i, (name, _, _, report) in enumerate(best_models):
        with columns[i % 3]:  # cycle through the columns
            st.markdown(f"<h4>{name}</h4>", unsafe_allow_html=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format({
                "precision": "{:.2f}%",
                "recall": "{:.2f}%",
                "f1-score": "{:.2f}%",
                "support": "{:,.0f}"
            }))

            st.markdown("<br>", unsafe_allow_html=True)  # just adds space

            # Prepare the CSV data for download
            classification_report_csv = pd.DataFrame(report).transpose().to_csv(index=False).encode('utf-8')
            file_name = f'{name}_classification_report.csv'
            mime='text/csv'

    st.markdown("<br>", unsafe_allow_html=True)  # just adds space

def display_model_comparison_plot(results, class_names):
    # Prep data for the plot
    comparison_data = []
    for name, acc, _, report in results:
        avg_precision = report['weighted avg']['precision']
        avg_recall = report['weighted avg']['recall']
        avg_f1_score = report['weighted avg']['f1-score']
        comparison_data.append([name, "Accuracy", acc])
        comparison_data.append([name, "Avg Precision", avg_precision])
        comparison_data.append([name, "Avg Recall", avg_recall])
        comparison_data.append([name, "Avg F1-Score", avg_f1_score])

    comparison_df = pd.DataFrame(comparison_data, columns=["Model", "Metric", "Score"])

    # Set a dark background and white text
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "black", "axes.edgecolor": "white"})

    # Create a bar plot with contrasting colors
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=comparison_df,
        x="Model",
        y="Score",
        hue="Metric",
        palette="rainbow",
    )
    
    # Add percentage labels on the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2%}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    color='white',
                    fontsize=6,
                    weight='bold')
    
    plt.title("Model Comparison Metrics", color="white", fontsize=16, pad=20)
    plt.xlabel("Model", color="white", fontsize=10)
    plt.ylabel("Score", color="white", fontsize=10)
    plt.ylim(0, 1)

    # Set the legend below the bar plot, above the model names
    plt.legend(loc="lower right", ncol=len(comparison_df['Metric'].unique()), facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)
    
    plt.xticks(rotation=45, color="white", fontsize=10)
    plt.yticks(color="white", fontsize=10)
    plt.tight_layout(pad=3.0)

    st.pyplot(plt)
    st.markdown("<br>", unsafe_allow_html=True)  # just adds space

# Function to calculate and plot learning curve
def plot_learning_curve(estimator, X_train, y_train, title="Learning Curve"):
    st.write(f"## {title}")
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, y_train, cv=5, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    # Calculate mean and standard deviation of scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.4,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.3,
        color="g",
    )
    ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    ax.set_xlabel("Training Examples", fontsize=8, color="white")
    ax.set_ylabel("Score", fontsize=9, color="white")
    ax.tick_params(axis='x', labelsize=8, colors='white')
    ax.tick_params(axis='y', labelsize=8, colors='white')
    ax.legend(loc="best", facecolor='lightgrey')
    ax.grid(True)
    st.pyplot(fig)

    description = f"""
    This figure shows the learning curve, which visualizes the performance of a machine learning model as it trains on more and more data. This helps understand the model's learning progress and can identify potential issues such as overfitting or underfitting.
    
    * If both the training score and cross-validation score are high and close to each other, it indicates that the model is performing well with no obvious signs of overfitting or underfitting.
    * If the training score is high but the cross-validation score is low, it suggests that the model is overfitting. This means the model is performing well on the training data but failing to generalize to new, unseen data.
    * If both the training score and cross-validation score are low, it suggests that the model is underfitting. This means the model is not performing well even on the training data and is too simple to capture the underlying patterns in the data.
    """

    st.write(description)

def run_algoDescription():
    with open('model.py', 'r', encoding='utf-8') as file:
        exec(file.read())

#########

# Default settings for vehicle types (approximation lang by the programmer). Can be changed.
default_settings = {
    "Car": {"Speed (km/h)": (100, 20), "Weight (kg)": (1500, 200), "Fuel Efficiency (km/l)": (15, 2)},
    "Truck": {"Speed (km/h)": (80, 15), "Weight (kg)": (5000, 500), "Fuel Efficiency (km/l)": (5, 1)},
    "Bike": {"Speed (km/h)": (60, 10), "Weight (kg)": (200, 50), "Fuel Efficiency (km/l)": (50, 5)},
    "Bus": {"Speed (km/h)": (70, 10), "Weight (kg)": (7000, 700), "Fuel Efficiency (km/l)": (4, 1)}
}

# Streamlit app
st.set_page_config(layout="wide")

# Add navigation options to the sidebar
sidebar = st.sidebar
sidebar.header("Navigation")
navigation = sidebar.radio("Go to", ["App", "Algorithm Education", "Model Implementation"])

# Main content based on navigation selection
if navigation == "App":
    st.title("Vehicle Type Classification App")
    st.markdown("""
    Configure the data on the sidebar to generate synthetic vehicle data, train multiple models, and compare their performance.
    """)

    # Sidebar: Data source configuration
    st.sidebar.header("Synthetic Data Configuration")

    # Input feature names
    feature_names = st.sidebar.text_input("Enter feature names (comma-separated)", "Speed (km/h), Weight (kg), Fuel Efficiency (km/l)")
    features = [f.strip() for f in feature_names.split(",")]

    # Input class names
    class_names = st.sidebar.text_input("Enter class names (comma-separated)", "Car, Truck, Bike, Bus")
    classes = [c.strip() for c in class_names.split(",")]

    # Class-specific settings
    st.sidebar.write("### Class-Specific Settings")
    class_settings = {}
    for class_name in classes:
        with st.sidebar.expander(f"{class_name} Settings", expanded=False):
            settings = {}
            for feature in features:
                default_mean, default_std_dev = default_settings.get(class_name, {}).get(feature, (50, 10))
                mean = st.number_input(f"Mean for {feature}", value=default_mean, key=f"mean_{feature}_{class_name}")
                std_dev = st.number_input(f"Std Dev for {feature}", value=default_std_dev, key=f"std_dev_{feature}_{class_name}")
                settings[feature] = (mean, std_dev)
            class_settings[class_name] = settings

    # Sample size and test size configuration
    st.sidebar.write("### Sample Size & Train/Test Split Configuration")
    total_samples = st.sidebar.number_input(
        "Enter the Number of Samples", min_value=100, max_value=50000, value=500, step=100
    )

    test_size_percentage = st.sidebar.number_input(
        "Enter the Test Size (%)", min_value=10, max_value=50, value=30, step=1
    ) / 100

    # Generate synthetic data and train models
    if st.sidebar.button("Generate Synthetic Data and Train Models"):
        try:
            # Generate data
            df, train_df, test_df = generate_vehicle_data(features, classes, total_samples, test_size_percentage, class_settings)
            st.session_state['df'] = df
            st.session_state['train_df'] = train_df
            st.session_state['test_df'] = test_df
            st.success("Synthetic data generated successfully!")

            # Train and compare models
            results, class_names, scaled_df, models = compare_models(train_df, test_df, features)
            st.session_state['results'] = results
            st.session_state['class_names'] = class_names
            st.session_state['scaled_df'] = scaled_df
            st.session_state['models'] = models

        except Exception as e:
            st.error(f"Error generating data or comparing models: {e}")

    # Use previously generated data if available
    if 'df' in st.session_state:
        df = st.session_state['df']
        train_df = st.session_state['train_df']
        test_df = st.session_state['test_df']
        scaled_df = st.session_state['scaled_df']
    else:
        # Default empty DataFrame if no data is available
        df, train_df, test_df, scaled_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Perform EDA and plot results if data is available
    if not df.empty:
        # Display dataset split information
        train_size = len(train_df)
        test_size_actual = len(test_df)
        train_percentage = int((train_size / total_samples) * 100)
        test_percentage = int((test_size_actual / total_samples) * 100)

        st.write("## Dataset Split Information")
        df_total_count, df_train_count, df_test_count = st.columns(3)
        df_total_count.metric(label="Total Samples", value=f"{total_samples}", delta=f"")
        df_train_count.metric("Training Samples", f"{train_size} ({train_percentage}%)")
        df_test_count.metric("Test Samples", f"{test_size_actual} ({test_percentage}%)")
        st.divider()

        # Perform EDA
        perform_eda(df, scaled_df, features)
        st.divider()

        # Plot results
        if 'results' in st.session_state and 'class_names' in st.session_state:
            plot_results(st.session_state['results'], st.session_state['class_names'])
            # Display model comparison table
            display_model_comparison_plot(st.session_state['results'], st.session_state['class_names'])
            # display learning curve
            st.write("## Learning Curves")
            
            # Access the models dictionary from session state
            models = st.session_state['models']
            
            # Select a model for the learning curve
            selected_model = st.selectbox("Select a Model to Plot Learning Curve", list(models.keys()))
            selected_model_instance = models[selected_model]
            
            # Refit the selected model on the training data
            X_train = train_df[features]
            y_train = train_df['Class']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            y_train_encoded = LabelEncoder().fit_transform(y_train)
            
            try:
                plot_learning_curve(selected_model_instance, X_train_scaled, y_train_encoded, title=f"Learning Curve for {selected_model}")
            except Exception as e:
                st.error(f"Error plotting learning curve: {e}")

elif navigation == "Algorithm Education":
    run_algoDescription()

elif navigation == "Model Implementation":
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File successfully uploaded!")
            st.write("### Uploaded Dataset")
            st.dataframe(df)
            
            # Define features
            features = [col for col in df.columns if col != 'Class']  # assuming 'Class' is the target column

            # Train and evaluate models using the uploaded dataset
            train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['Class'], random_state=42)
            results, class_names, scaled_df, models = compare_models(train_df, test_df, features)

            # Store the data in session state
            st.session_state['df'] = df
            st.session_state['train_df'] = train_df
            st.session_state['test_df'] = test_df
            st.session_state['results'] = results
            st.session_state['class_names'] = class_names
            st.session_state['scaled_df'] = scaled_df
            st.session_state['models'] = models

            # Perform EDA
            perform_eda(df, scaled_df, features)
            st.divider()

            # Plot results
            plot_results(results, class_names)
            display_model_comparison_plot(results, class_names)

            # Display learning curve
            st.write("## Learning Curves")
            selected_model = st.selectbox("Select a Model to Plot Learning Curve", list(models.keys()))
            selected_model_instance = models[selected_model]

            # Refit the selected model on the training data
            X_train = train_df[features]
            y_train = train_df['Class']

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            y_train_encoded = LabelEncoder().fit_transform(y_train)

            plot_learning_curve(selected_model_instance, X_train_scaled, y_train_encoded, title=f"Learning Curve for {selected_model}")

        except Exception as e:
            st.error(f"Error processing file: {e}")
