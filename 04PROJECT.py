import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# ===============================
# 1. Load Data
# ===============================
@st.cache_data
def load_data():
    base_path = "D:/GUVI/MINI PROJECT/GUVI-Prj04/Tourism Dataset"
    try:
        datasets = {
            "city": pd.read_excel(f"{base_path}/City.xlsx"),
            "continent": pd.read_excel(f"{base_path}/Continent.xlsx"),
            "country": pd.read_excel(f"{base_path}/Country.xlsx"),
            "item": pd.read_excel(f"{base_path}/Item.xlsx"),
            "mode": pd.read_excel(f"{base_path}/Mode.xlsx"),
            "region": pd.read_excel(f"{base_path}/Region.xlsx"),
            "transaction": pd.read_excel(f"{base_path}/Transaction.xlsx"),
            "type_df": pd.read_excel(f"{base_path}/Type.xlsx"),
            "user": pd.read_excel(f"{base_path}/User.xlsx")
        }
        return datasets
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ===============================
# 2. Preprocess Data
# ===============================
@st.cache_data
def preprocess_data(transaction, user, item):
    for df in [transaction, user, item]:
        df.columns = df.columns.str.strip()
    transaction.rename(columns={"UserId": "Userid","AttractionId": "Itemid","VisitYear": "Year","VisitMonth": "Month"}, inplace=True)
    user.rename(columns={"UserId": "Userid","ContinentId": "Continent","RegionId": "Region","CountryId": "Country","CityId": "City"}, inplace=True)
    item.rename(columns={"AttractionId": "Itemid","AttractionTypeId": "Type"}, inplace=True)
    df = transaction.merge(user, on='Userid', how='left').merge(item, on='Itemid', how='left')
    df.dropna(inplace=True)
    le = LabelEncoder()
    for col in ['VisitMode','Continent','Country','Region','City','Type']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# ===============================
# 3. Filter by City and Attraction
# ===============================
def filter_data(df, city_df, selected_city, selected_attraction):
    city_row = city_df[city_df['CityName'] == selected_city]
    if city_row.empty:
        st.warning(f"City {selected_city} not found.")
        return pd.DataFrame()
    city_id = city_row['CityId'].values[0]
    filtered = df[df['City'] == city_id]
    if selected_attraction != "All":
        filtered = filtered[filtered['Attraction'] == selected_attraction]
    if filtered.empty:
        st.warning("No data for selected city and attraction.")
    return filtered

# ===============================
# 4. EDA Visualization
# ===============================
def eda_visualizations(df):
    st.markdown("## 🔎 Tourism Analytics Overview")
    if df.empty:
        st.info("No data to display.")
        return
    if 'VisitMode' in df.columns:
        st.subheader("Visit Mode distribution")
        st.bar_chart(df['VisitMode'].value_counts())
    if 'Type' in df.columns:
        st.subheader("Average Rating by Attraction Type")
        st.bar_chart(df.groupby('Type')['Rating'].mean())
    if 'Rating' in df.columns:
        st.subheader("Ratings distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Rating'], bins=10, ax=ax)
        st.pyplot(fig)

# ===============================
# 5. Run Regression
# ===============================
def regression_module(df):
    st.markdown("## 📈 Predict Attraction Ratings")
    if df.empty:
        st.info("No data for regression.")
        return
    features = ['Userid','Itemid','Year','Month','Continent','Region','Country','City','VisitMode','Type']
    target = 'Rating'
    if not all(col in df.columns for col in features + [target]):
        st.warning("Missing required columns.")
        return
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.metric("R² Score", f"{r2_score(y_test, preds):.2f}")
    st.metric("MSE", f"{mean_squared_error(y_test, preds):.2f}")
    fig, ax = plt.subplots()
    sns.histplot(y_test-preds, kde=True, ax=ax)
    ax.set_title("Residuals of Rating Prediction")
    st.pyplot(fig)

# ===============================
# 6. Run Classification
# ===============================
def classification_module(df):
    st.markdown("## 🧠 Predict Visit Mode")
    if df.empty:
        st.info("No data for classification.")
        return
    features = ['Userid','Itemid','Year','Month','Continent','Region','Country','City','Type']
    target = 'VisitMode'
    if not all(col in df.columns for col in features + [target]):
        st.warning("Missing required columns.")
        return
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2f}")
    st.metric("F1 Score", f"{f1_score(y_test, preds, average='weighted'):.2f}")
    st.text("Classification Report:\n" + classification_report(y_test, preds))
    fig, ax = plt.subplots()
    sns.countplot(x=preds, ax=ax)
    ax.set_title("Predicted Visit Modes")
    st.pyplot(fig)

# ===============================
# 7. Recommendations
# ===============================
def recommendation_module(df):
    st.markdown("## 🎯 Recommendations")
    if df.empty:
        st.info("No data for recommendations.")
        return
    user_item = df.pivot_table(index='Userid', columns='Itemid', values='Rating').fillna(0)
    user_id = st.selectbox("Select User ID:", sorted(df['Userid'].unique()))
    if st.button("Show Recommendations"):
        if user_id not in user_item.index:
            st.warning("User ID not found.")
            return
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(user_item)
        distances, indices = model.kneighbors(user_item.loc[[user_id]], n_neighbors=3)
        neighbors = user_item.iloc[indices[0]].index.tolist()
        rec_df = df[df['Userid'].isin(neighbors) & (df['Userid'] != user_id)]
        top_items = rec_df.groupby('Itemid')['Rating'].mean().sort_values(ascending=False).head(5)
        st.table(top_items.reset_index().rename(columns={'Itemid':'Attraction ID','Rating':'Avg. Rating'}))

# ===============================
# 8. Main App
# ===============================
def main():
    st.title("🌍 Tourism Experience Analytics")

    data = load_data()
    if data is None:
        st.stop()

    city_df = data['city']
    df = preprocess_data(data['transaction'], data['user'], data['item'])
    if df.empty:
        st.error("No data after preprocessing.")
        st.stop()

    # City selection
    city_list = city_df['CityName'].sort_values().unique()
    selected_city = st.selectbox("Select City:", city_list)

    # Attractions in selected city
    city_id = city_df[city_df['CityName'] == selected_city]['CityId'].values[0]
    attractions = df[df['City'] == city_id]['Attraction'].unique()
    attraction_options = ["All"] + list(sorted(attractions))
    selected_attraction = st.selectbox("Select Attraction:", attraction_options)

    filtered_df = filter_data(df, city_df, selected_city, selected_attraction)

    st.markdown(f"### Data Preview for {selected_city} - {selected_attraction}")
    with st.expander("Show Raw Data"):
        st.dataframe(filtered_df.head(20))

    eda_visualizations(filtered_df)
    st.markdown("---")
    regression_module(filtered_df)
    st.markdown("---")
    classification_module(filtered_df)
    st.markdown("---")
    recommendation_module(filtered_df)

if __name__ == '__main__':
    main()
