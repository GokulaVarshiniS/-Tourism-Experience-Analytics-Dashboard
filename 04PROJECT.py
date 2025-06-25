import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# ==========================
# Data Loading
# ==========================
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

# ==========================
# Data Preprocessing
# ==========================
@st.cache_data
def preprocess_data(transaction, user, item):
    try:
        for df in [transaction, user, item]:
            df.columns = df.columns.str.strip()

        transaction.rename(columns={"UserId": "Userid", "AttractionId": "Itemid", "VisitYear": "Year", "VisitMonth": "Month"}, inplace=True)
        user.rename(columns={"UserId": "Userid", "ContinentId": "Continent", "RegionId": "Region", "CountryId": "Country", "CityId": "City"}, inplace=True)
        item.rename(columns={"AttractionId": "Itemid", "AttractionTypeId": "Type"}, inplace=True)

        df = transaction.merge(user, on='Userid', how='left').merge(item, on='Itemid', how='left')
        df.dropna(inplace=True)

        le = LabelEncoder()
        for col in ['VisitMode', 'Continent', 'Country', 'Region', 'City', 'Type']:
            if col in df.columns:
                df[col] = le.fit_transform(df[col].astype(str))

        return df
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return pd.DataFrame()

# ==========================
# Regression
# ==========================
def run_regression(df):
    st.subheader("📈 Predict Ratings (Regression)")
    features = ['Userid', 'Itemid', 'Year', 'Month', 'Continent', 'Region', 'Country', 'City', 'VisitMode', 'Type']
    target = 'Rating'

    if not all(col in df.columns for col in features + [target]):
        st.warning("Required columns missing for regression.")
        return

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    st.metric("R² Score", f"{r2_score(y_test, predictions):.2f}")
    st.metric("MSE", f"{mean_squared_error(y_test, predictions):.2f}")

# ==========================
# Classification
# ==========================
def run_classification(df):
    st.subheader("🧠 Predict Visit Mode (Classification)")
    features = ['Userid', 'Itemid', 'Year', 'Month', 'Continent', 'Region', 'Country', 'City', 'Type']
    target = 'VisitMode'

    if not all(col in df.columns for col in features + [target]):
        st.warning("Required columns missing for classification.")
        return

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    st.text(classification_report(y_test, predictions))

# ==========================
# Recommendation
# ==========================
def recommend_attractions(df, user_id):
    st.subheader("🎯 Attraction Recommendations")

    if 'Userid' not in df.columns or 'Itemid' not in df.columns or 'Rating' not in df.columns:
        st.warning("Missing required columns for recommendation.")
        return

    user_item = df.pivot_table(index='Userid', columns='Itemid', values='Rating').fillna(0)
    if user_id not in user_item.index:
        st.warning("User ID not found in the dataset.")
        return

    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item)
    distances, indices = model.kneighbors(user_item.loc[[user_id]], n_neighbors=3)

    neighbors = user_item.iloc[indices[0]].index.tolist()
    recommended = df[df['Userid'].isin(neighbors) & (df['Userid'] != user_id)]
    top_items = recommended.groupby('Itemid')['Rating'].mean().sort_values(ascending=False).head(5)
    st.write("Top 5 Recommended Attractions:")
    st.dataframe(top_items.reset_index())

# ==========================
# Main Streamlit App
# ==========================
def main():
    st.title("🌍 Tourism Experience Analytics Dashboard")

    data = load_data()
    if data is None:
        return

    city, continent, country, item, mode, region, transaction, type_df, user = [data[k] for k in data]
    df = preprocess_data(transaction, user, item)

    if df.empty:
        return

    st.write("### Preview of Data", df.head())
    run_regression(df)
    run_classification(df)

    valid_user_ids = sorted(df['Userid'].unique())
    user_id = st.selectbox("🔍 Select a User ID for Recommendations", valid_user_ids)
    if st.button("Recommend Attractions"):
        recommend_attractions(df, user_id)

if __name__ == '__main__':
    main()
