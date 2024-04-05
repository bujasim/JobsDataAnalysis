import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import clipboard
# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Set page layout to wide mode
st.set_page_config(layout="wide")

# Update the caching command based on Streamlit's latest practices
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

df = None
# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Format date fields or any other preprocessing
    df['job_posting_date'] = pd.to_datetime(df['job_posting_date'], format='%Y-%m-%dT%H:%M:%S.%f%z', errors='coerce').dt.strftime('%B %d, %Y %H:%M:%S')

skills_and_compentencies_column = 'skills_competencies'
cultural_environmental_column = 'cultural_environmental'
professional_interests_goals_column = 'professional_interests_goals'
work_type_flexibility_column = 'work_type_flexibility'
professional_experiences_achievements_column = 'professional_experiences_achievements'


# Define the columns that contain feature scores
feature_score_columns = [
    f'{skills_and_compentencies_column}_positive',
    f'{cultural_environmental_column}_positive',
    f'{professional_interests_goals_column}_positive',
    f'{work_type_flexibility_column}_positive',
    f'{professional_experiences_achievements_column}_positive',
    f'{skills_and_compentencies_column}_negative',
    f'{cultural_environmental_column}_negative',
    f'{professional_interests_goals_column}_negative',
    f'{work_type_flexibility_column}_negative',
    f'{professional_experiences_achievements_column}_negative',
    'total_weighted_score'  # Add 'total_weighted_score' to the list
]

# Define a dictionary that maps the original column names to the aliases
aliases = {
    f'{skills_and_compentencies_column}_positive': 'Skills & Competencies (+)',
    f'{cultural_environmental_column}_positive': 'Cultural & Environmental (+)',
    f'{professional_interests_goals_column}_positive': 'Professional Interests & Goals (+)',
    f'{work_type_flexibility_column}_positive': 'Work Type Flexibility (+)',
    f'{professional_experiences_achievements_column}_positive': 'Professional Experiences & Achievements (+)',
    f'{skills_and_compentencies_column}_negative': 'Skills & Competencies (-)',
    f'{cultural_environmental_column}_negative': 'Cultural & Environmental (-)',
    f'{professional_interests_goals_column}_negative': 'Professional Interests & Goals (-)',
    f'{work_type_flexibility_column}_negative': 'Work Type Flexibility (-)',
    f'{professional_experiences_achievements_column}_negative': 'Professional Experiences & Achievements (-)',
    'total_weighted_score': 'Total Weighted Score'  # Add an alias for 'total_weighted_score'
}


# Initialize session state for job index if it doesn't exist
if 'job_index' not in st.session_state:
    st.session_state.job_index = 0
if df is not None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    # Create a copy of the DataFrame
    df_copy = df.copy()
    # Convert 'job_industry' column to string
    df_copy['job_industry'] = df_copy['job_industry'].astype(str)

    # Check if the model and transformed data already exist in the session state
    if 'model' not in st.session_state or 'X' not in st.session_state:
        # Vectorize the text data
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df_copy['job_industry'])

        # Fit the KMeans model
        model = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1, random_state=42)
        model.fit(X)

        # Store the model and transformed data in the session state
        st.session_state.model = model
        st.session_state.X = X
    else:
        # Retrieve the model and transformed data from the session state
        model = st.session_state.model
        X = st.session_state.X

    # Add the cluster labels to the copied DataFrame
    df_copy['industry_cluster'] = model.labels_

    industry_scores = df_copy.groupby('industry_cluster')[feature_score_columns].mean()


    #######################################################

    # Get the top 3 'job_industry' values in each cluster
    top_industries = df_copy.groupby('industry_cluster')['job_industry'].apply(lambda x: x.value_counts().index[:3])

    # Create three columns
    col1, col2 = st.columns([4,2])

    def set_chart_style(ax):
        fig.patch.set_facecolor('#323C52')  # Set figure background color
        ax.set_facecolor('#191E29')  # Set chart area background color
        ax.set_xlabel('Scores', color='white', fontsize=10, weight='bold')  # Double the font size
        ax.tick_params(axis='both', colors='white', labelsize=10)  # Double the font size of the tick labels
        ax.xaxis.label.set_color('white')  # Set color of x-axis label to white
        ax.yaxis.label.set_color('white')  # Set color of y-axis label to white


    with col1:
        st.write("Top industries in each cluster:")
        for cluster, industries in top_industries.items():
            st.write(f"Cluster {cluster}: {', '.join(industries)}")

    # Create a smaller figure and axes in the first column
    with col2:
        fig, axes = plt.subplots(figsize=(6, 4))  # Adjust the numbers as needed

        # Plot industry scores
        sns.barplot(ax=axes, x=industry_scores.index, y=industry_scores['total_weighted_score'])
        axes.set_title('Scores by Industry Cluster')
        axes.tick_params(axis='x', rotation=90)

        # Set chart style
        set_chart_style(axes)

        # Display the plot in Streamlit
        st.pyplot(fig)


    #######################################################



    # Convert 'job_function' column to string
    df_copy['job_function'] = df_copy['job_function'].astype(str)

    # Check if the model and transformed data already exist in the session state
    if 'model_function' not in st.session_state or 'X_function' not in st.session_state:
        # Vectorize the text data
        vectorizer_function = TfidfVectorizer(stop_words='english')
        X_function = vectorizer_function.fit_transform(df_copy['job_function'])

        # Fit the KMeans model
        model_function = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1, random_state=42)
        model_function.fit(X_function)

        # Store the model and transformed data in the session state
        st.session_state.model_function = model_function
        st.session_state.X_function = X_function
    else:
        # Retrieve the model and transformed data from the session state
        model_function = st.session_state.model_function
        X_function = st.session_state.X_function

    # Add the cluster labels to the copied DataFrame
    df_copy['function_cluster'] = model_function.labels_

    function_scores = df_copy.groupby('function_cluster')[feature_score_columns].mean()

    # Get the top 3 'job_function' values in each cluster
    top_functions = df_copy.groupby('function_cluster')['job_function'].apply(lambda x: x.value_counts().index[:3])

    # Create three columns
    col1, col2 = st.columns([4,2])

    with col1:
        st.write("Top functions in each cluster:")
        for cluster, functions in top_functions.items():
            st.write(f"Cluster {cluster}: {', '.join(functions)}")

    # Create a smaller figure and axes in the first column
    with col2:
        fig, axes = plt.subplots(figsize=(6, 4))  # Adjust the numbers as needed

        # Plot function scores
        sns.barplot(ax=axes, x=function_scores.index, y=function_scores['total_weighted_score'])
        axes.set_title('Scores by Function Cluster')
        axes.tick_params(axis='x', rotation=90)

        # Set chart style
        set_chart_style(axes)

        # Display the plot in Streamlit
        st.pyplot(fig)
















#######################################################
def display_job_details(index):
    global df
    if df is not None:
        # Sort the dataframe by 'total_weighted_score' in descending order
        df = df.sort_values(by='total_weighted_score', ascending=False)

        # Make a "Rank" column starting from 1
        df['Rank'] = range(1, len(df) + 1)

        job = df.iloc[index]

        ########################################################
        # Navigation to browse through job profiles
        col1, col2, col3 = st.columns([1, 10, 1])  # Create three columns
        with col1:
            st.write("")  # This is to push the button down to align with the dataframe
            if st.button("Previous"):
                if st.session_state.job_index > 0:
                    st.session_state.job_index -= 1

        with col3:
            st.write("")  # This is to push the button down to align with the dataframe
            if st.button("Next"):
                if st.session_state.job_index < len(df) - 1:
                    st.session_state.job_index += 1

        # Display the dataframe in the middle column
        with st.container():
            # To highlight the current row in the DataFrame and display a few rows around it
            def display_dataframe_with_highlight(df, current_index):
                start_index = max(0, current_index - 2)  # Show 2 rows above the current row
                end_index = min(len(df), current_index + 3)  # Show 2 rows below the current row

                # Highlighting the current row
                def highlight_row(s, current_index=current_index, start_index=start_index):
                    return ['background-color: #323C52' if i == current_index else '' for i in range(start_index, end_index)]

                # Display the DataFrame slice with highlighted current row
                st.dataframe(df.iloc[start_index:end_index].style.apply(highlight_row, axis=0))

            # Insert a call to this function at the appropriate place in your code
            display_dataframe_with_highlight(df, st.session_state.job_index)

        ########################################################
        # Splitting the layout into two columns
        col1, col2 = st.columns([2,1])


        with col1:
            st.markdown("# Job Details")
            st.markdown("## " + job['job_position'])
            st.markdown("#### Job Posting Rank: # " + str(job['Rank']))
            st.markdown("**Company:** " + job['company_name'])
            st.markdown("**Location:** " + job['job_location'])
            st.markdown("**Type:** " + job['job_work_type'])
            st.markdown("**Seniority Level:** " + job['job_seniority_level'])
            st.markdown("**Posted:** " + job['job_posting_date'])
            st.markdown("**Job Posting URL:** " + "https://www.linkedin.com/jobs/view/" + str(job['job_posting_id']) + "/")

            st.markdown("## Description:")
            # Check if the 'show_more' key exists in the session state
            if 'show_more' not in st.session_state:
                st.session_state.show_more = False  # Initialize it to False
            # If 'show_more' is False, show truncated text and a "Show More" button
            if not st.session_state.show_more:
                # Using markdown to preserve newlines by replacing them with Markdown line breaks
                st.markdown(job['job_description'][:500].replace('\n', '  \n') + '...', unsafe_allow_html=True)  # Truncate for brevity
                if st.button('Show More'):
                    st.session_state.show_more = True  # Set 'show_more' to True when button is clicked
            # If 'show_more' is True, show full text and a "Show Less" button
            else:
                # Using markdown to preserve newlines by replacing them with Markdown line breaks
                st.markdown(job['job_description'].replace('\n', '  \n\n'), unsafe_allow_html=True)
                if st.button('Show Less'):
                    st.session_state.show_more = False  # Set 'show_more' to False when button is clicked

        with col2:
            st.header("Scores")
            
            fig, ax = plt.subplots(figsize=(6, 4))  # Set figure size
            
            fig.patch.set_facecolor('#323C52')  # Set figure background color
            
            job_scores = job[feature_score_columns]
            
            job_scores.index = [aliases[col] for col in job_scores.index]

            threshold_positive_value = 0.4
            threshold_negative_value = 0.4
            colors = []
            for col, score in job_scores.items():
                if '(+)' in col:
                    if score >= threshold_positive_value:
                        colors.append('#82BB5B') # Brightgreen
                    else:
                        colors.append('#B4D69C')  # Halfgreen
                elif '(-)' in col:
                    if score <= threshold_negative_value:
                        colors.append('#FBD3D4') # Brightred
                    else:
                        colors.append('#F08085')  # Halfred
                else:  # For 'Total Weighted Score'
                    colors.append('Gold')

            bars = ax.barh(job_scores.index, job_scores.values, color=colors)

            set_chart_style(ax)

            bars = ax.barh(job_scores.index, job_scores.values, color=colors)
            ax.set_xlim(-1, 1)  # Set x-axis range from 0 to 1
 
            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    f'{bar.get_width():.2f}', 
                    va='center', ha='left', 
                    color='white', fontsize=10, weight='bold')  # Double the font size
            plt.tight_layout()
            st.pyplot(fig)  # Display the plot
        #######################################################################################################

        # Generate job details string for copy feature
        job_details_string = f"Position: {job['job_position']}, Company: {job['company_name']}, Location: {job['job_location']}, Industry: {job['job_industry']}, Function: {job['job_function']}, Work Type: {job['job_work_type']}, Seniority Level: {job['job_seniority_level']}, Employment Type: {job['job_employment_type']}, Description: {job['job_description']}, Salary: {job['job_salary']}, Source: {job['job_posting_source']}"



        # Inside your display_job_details function, after generating the job_details_string
        def copy_to_clipboard(job_details):
            """
            Copies the provided job details to the clipboard.
            """
            clipboard.copy(job_details)
            st.session_state['copied'] = job_details  # Optional: Track the last copied job details

        # Add an initialization for the copied state if it does not exist
        if 'copied' not in st.session_state:
            st.session_state['copied'] = ""

        # Create a button that, when clicked, copies the job details to the clipboard
        if st.button("Copy Job Details to Clipboard"):
            copy_to_clipboard(job_details_string)
            st.success("Copied to clipboard!")  # Give user feedback

        # Optional: Display a toast or message if something was recently copied
        if st.session_state['copied']:
            st.toast(f"Copied to clipboard: {st.session_state['copied']}", icon='✅')







# Display the job details for the current index
display_job_details(st.session_state.job_index)
