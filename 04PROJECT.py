import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from scipy.stats import skew

# -------------------- Utility Functions --------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=False)
def load_data(base_path):
    """Load Excel files from the dataset folder"""
    files = ["City", "Continent", "Country", "Item", "Transaction", "User", "Mode", "Region"]
    data = {}
    for f in files:
        df = pd.read_excel(f"{base_path}/{f}.xlsx")
        df.columns = df.columns.str.strip()
        data[f.lower()] = df
    return data

@st.cache_data(show_spinner=False)
def preprocess(data):
    """Merge datasets and replace IDs with readable names"""
    city, continent, country, item, transaction, user, mode, region = (
        data["city"], data["continent"], data["country"], data["item"],
        data["transaction"], data["user"], data["mode"], data["region"]
    )

    visitmode_col = find_col(transaction, ["VisitModeId", "VisitMode", "ModeId", "Mode"])
    if not visitmode_col:
        st.error("❌ VisitMode column not found in Transaction.xlsx")
        st.stop()

    # Rename columns
    transaction.rename(columns={
        "UserId": "Userid", "AttractionId": "Itemid",
        "VisitYear": "Year", "VisitMonth": "Month", visitmode_col: "VisitModeId"
    }, inplace=True)
    user.rename(columns={"UserId": "Userid"}, inplace=True)
    item.rename(columns={"AttractionId": "Itemid", "AttractionTypeId": "Type"}, inplace=True)

    df = (transaction
        .merge(user, on='Userid', how='left')
        .merge(item, on='Itemid', how='left')
        .merge(mode, on='VisitModeId', how='left')
        .merge(city[['CityId', find_col(city, ["City", "CityName"])]], on='CityId', how='left')
        .merge(region[['RegionId', find_col(region, ["Region", "RegionName"])]], on='RegionId', how='left')
        .merge(country[['CountryId', find_col(country, ["Country", "CountryName"])]], on='CountryId', how='left')
        .merge(continent[['ContinentId', find_col(continent, ["Continent", "ContinentName"])]], on='ContinentId', how='left')
    )

    df.rename(columns={
        find_col(continent, ["Continent", "ContinentName"]): 'Continent',
        find_col(country, ["Country", "CountryName"]): 'Country',
        find_col(city, ["City", "CityName"]): 'City',
        find_col(region, ["Region", "RegionName"]): 'Region',
        'ModeName': 'VisitMode'
    }, inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                s = skew(df[col].dropna())
                df[col].fillna(df[col].median() if abs(s) > 1 else df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# -------------------- Cached Model Training (Optimized) --------------------
@st.cache_resource(show_spinner=False)
def get_models(df):
    """Train models (on a smaller sample) and select the best one"""
    reg_features = ['Year', 'Month', 'Continent', 'Region', 'Country', 'City', 'VisitMode', 'Type']
    cls_features = ['Year', 'Month', 'Continent', 'Region', 'Country', 'City', 'Type']

    best_reg_model, best_cls_model = None, None
    reg_summary, cls_summary = None, None

    # -------------------- Regression --------------------
    if 'Rating' in df.columns:
        reg_data = df.dropna(subset=reg_features + ['Rating'])
        if not reg_data.empty:
            reg_data = reg_data.sample(min(500, len(reg_data)), random_state=42)  # Limit sample for speed
            Xr = pd.get_dummies(reg_data[reg_features])
            yr = reg_data['Rating']

            models = {
                "RandomForestRegressor": RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                "LinearRegression": LinearRegression()
            }

            scores = {}
            for name, model in models.items():
                model.fit(Xr, yr)
                pred = model.predict(Xr)
                scores[name] = r2_score(yr, pred)

            reg_summary = pd.DataFrame({"Model": list(scores.keys()), "R2_Score": list(scores.values())})
            best_model_name = reg_summary.loc[reg_summary['R2_Score'].idxmax(), 'Model']
            best_reg_model = models[best_model_name]

    # -------------------- Classification --------------------
    if 'VisitMode' in df.columns:
        cls_data = df.dropna(subset=cls_features + ['VisitMode'])
        if not cls_data.empty:
            cls_data = cls_data.sample(min(500, len(cls_data)), random_state=42)
            Xc = pd.get_dummies(cls_data[cls_features])
            yc = cls_data['VisitMode']

            models = {
                "RandomForestClassifier": RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1),
                "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=500)
            }

            accs = {}
            for name, model in models.items():
                model.fit(Xc, yc)
                pred = model.predict(Xc)
                accs[name] = accuracy_score(yc, pred)

            cls_summary = pd.DataFrame({"Model": list(accs.keys()), "Accuracy": list(accs.values())})
            best_cls_name = cls_summary.loc[cls_summary['Accuracy'].idxmax(), 'Model']
            best_cls_model = models[best_cls_name]

    return best_reg_model, best_cls_model, reg_features, cls_features, reg_summary, cls_summary

# -------------------- Helper --------------------
def show_selectors(df, features):
    user_input = {}
    for col in features:
        vals = sorted(df[col].dropna().astype(str).unique())
        user_input[col] = st.selectbox(f"Select {col}", ["All"] + vals)
    return user_input

def resolve_all_inputs(user_input, data):
    for col in user_input:
        if user_input[col] == "All":
            user_input[col] = data[col].mode()[0]
    return user_input

# -------------------- Main App --------------------
def main():
    st.set_page_config(page_title="🌍 Tourism Experience Analytics", layout="wide")
    st.title("🌍 Tourism Experience Analytics System")

    base_path = "D:/GUVI/MINI PROJECT/GUVI-Prj04/Tourism Dataset"
    with st.spinner("Loading and preparing data..."):
        data = load_data(base_path)
        df = preprocess(data)
        reg_model, cls_model, reg_features, cls_features, reg_summary, cls_summary = get_models(df)
    st.success("✅ Data and models ready!")

    module = st.radio("Choose Module", ["Regression", "Classification", "Recommendation"])

    # -------------------- REGRESSION --------------------
    if module == "Regression":
        st.header("📈 Tourist Rating Prediction")
        if reg_model is None:
            st.error("❌ No regression data available.")
            return

        st.subheader("📊 Model Comparison")
        st.dataframe(reg_summary)
        best_model = reg_summary.loc[reg_summary['R2_Score'].idxmax(), 'Model']
        st.info(f"🏆 Best Regression Model: **{best_model}**")

        user_input = show_selectors(df, reg_features)
        if st.button("🔮 Predict Rating"):
            user_input = resolve_all_inputs(user_input, df)
            input_df = pd.DataFrame([user_input])
            input_X = pd.get_dummies(input_df).reindex(columns=reg_model.feature_names_in_, fill_value=0)
            pred = reg_model.predict(input_X)[0]
            st.success(f"🌟 Predicted Tourist Rating: {pred:.2f} / 5")

    # -------------------- CLASSIFICATION --------------------
    elif module == "Classification":
        st.header("🧭 Visit Mode Prediction")
        if cls_model is None:
            st.error("❌ No classification data available.")
            return

        st.subheader("📊 Model Comparison")
        st.dataframe(cls_summary)
        best_model = cls_summary.loc[cls_summary['Accuracy'].idxmax(), 'Model']
        st.info(f"🏆 Best Classification Model: **{best_model}**")

        user_input = show_selectors(df, cls_features)
        if st.button("🎯 Predict Visit Mode"):
            user_input = resolve_all_inputs(user_input, df)
            input_df = pd.DataFrame([user_input])
            input_X = pd.get_dummies(input_df).reindex(columns=cls_model.feature_names_in_, fill_value=0)
            pred = cls_model.predict(input_X)[0]
            st.success(f"🚗 Predicted Visit Mode: **{pred}**")

    # -------------------- RECOMMENDATION --------------------
    else:
        st.header("💡 Attraction Recommendations")
        features = ['Year', 'Month', 'Continent', 'Region', 'Country', 'City', 'VisitMode', 'Type']
        if 'Rating' not in df.columns:
            st.warning("Rating column missing for recommendation.")
            return
        data = df.dropna(subset=features)
        user_input = show_selectors(data, features)
        if st.button("✨ Get Recommendations"):
            filtered_data = data.copy()
            for col, val in user_input.items():
                if val != "All":
                    filtered_data = filtered_data[filtered_data[col].astype(str) == val]

            if filtered_data.empty:
                st.warning("No attractions found matching your selections.")
                return

            showcols = ['Attraction', 'Continent', 'Country', 'Region', 'City', 'VisitMode', 'Type', 'Rating']
            top10 = filtered_data.sort_values('Rating', ascending=False).head(10)
            st.success(f"✨ Showing top recommendations for your selected filters.")
            st.dataframe(top10[showcols])

if __name__ == "__main__":
    main()
