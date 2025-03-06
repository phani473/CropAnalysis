import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load Model & Scaler
class CropDataProcessor:
    def __init__(self, data_path, model_path, scaler_path):
        self.data_path = data_path
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.df = self.load_and_preprocess_data()
    
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        df.dropna(inplace=True)
        df = df[['Area', 'Item', 'Year', 'Element', 'Value']]
        df = df.pivot_table(index=['Area', 'Item', 'Year'], columns='Element', values='Value').reset_index()
        
        # Ensure columns exist before renaming
        if 'Area harvested' in df.columns:
            df.rename(columns={'Area harvested': 'Area_Harvested'}, inplace=True)
        if 'Production' in df.columns:
            df.rename(columns={'Production': 'Production'}, inplace=True)
        
        df = df[['Area', 'Item', 'Year', 'Area_Harvested', 'Production']].dropna()
        
        # Calculate Yield
        if 'Production' in df.columns and 'Area_Harvested' in df.columns:
            df['Yield'] = df['Production'] / df['Area_Harvested']
        
        return df
    
data_processor = CropDataProcessor('FAOSTAT_data.csv', 'crop_model.pkl', 'scaler.pkl')
df = data_processor.df

st.title("Crop Production Analysis & Prediction")

# **1️⃣ Crop Type Distribution**
st.subheader("1. Crop Distribution Analysis")

# Count of crops by frequency
crop_counts = df['Item'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=crop_counts.index[:10], y=crop_counts.values[:10], ax=ax)
ax.set_title("Top 10 Most Cultivated Crops")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=crop_counts.index[-10:], y=crop_counts.values[-10:], ax=ax, color="red")
ax.set_title("Bottom 10 Least Cultivated Crops")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# **2️⃣ Geographical Distribution**
st.subheader("Geographical Crop Distribution")

# Select crop to see distribution
selected_crop = st.selectbox("Select Crop", df['Item'].unique())

crop_area_counts = df[df['Item'] == selected_crop]['Area'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=crop_area_counts.index[:10], y=crop_area_counts.values[:10], ax=ax)
ax.set_title(f"Top 10 Regions for {selected_crop}")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# **3️⃣ Temporal Analysis - Yearly Trends**

st.header("2.Temporal Analysis")
st.subheader("2.1 📈 Yearly Trends in Agriculture")

# Aggregate yearly data
yearly_trends = df.groupby("Year")[["Area_Harvested", "Yield", "Production"]].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=yearly_trends, x="Year", y="Area_Harvested", label="Area Harvested", ax=ax)
sns.lineplot(data=yearly_trends, x="Year", y="Yield", label="Yield", ax=ax)
sns.lineplot(data=yearly_trends, x="Year", y="Production", label="Production", ax=ax)
ax.set_title("Yearly Trends: Area Harvested, Yield, and Production")
ax.set_ylabel("Value")
st.pyplot(fig)

# **4️⃣ Growth Analysis - Crop/Region Trends**
st.subheader("2.2 📊 Growth Analysis")

# User selects crop or region
analysis_type = st.radio("Analyze Growth by:", ["Item", "Area"])
selected = st.selectbox(f"Select {analysis_type}", df[analysis_type].unique())

# Filter data
if analysis_type == "Item":
    trend_data = df[df["Item"] == selected].groupby("Year")[["Yield", "Production"]].mean().reset_index()
else:
    trend_data = df[df["Area"] == selected].groupby("Year")[["Yield", "Production"]].mean().reset_index()

# Plot trends
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=trend_data, x="Year", y="Yield", label="Yield", ax=ax)
sns.lineplot(data=trend_data, x="Year", y="Production", label="Production", ax=ax)
ax.set_title(f"Yearly Growth Trend for {selected}")
ax.set_ylabel("Value")
st.pyplot(fig)

st.subheader("3.🌱 Environmental Relationships")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=df, x="Area_Harvested", y="Yield", alpha=0.5)
ax.set_title("Impact of Area Harvested on Yield")
ax.set_xlabel("Area Harvested")
ax.set_ylabel("Yield")
st.pyplot(fig)

st.subheader("4.🔄 Input-Output Relationships")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df[["Area_Harvested", "Yield", "Production"]].corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Correlation Between Area Harvested, Yield, and Production")
st.pyplot(fig)


st.subheader("5.📊 Comparative Analysis")

# Compare yields across crops
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="Item", y="Yield")
ax.set_title("5.1 Yield Comparison Across Crops")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# Compare production across regions
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="Area", y="Production")
ax.set_title("5.2 Production Comparison Across Regions")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# **8️⃣ Productivity Analysis**
st.subheader("5.3 📈 Productivity Analysis")

df["Productivity_Ratio"] = df["Production"] / df["Area_Harvested"]
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="Item", y="Productivity_Ratio")
ax.set_title("Productivity Ratios Across Crops")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)


st.subheader("6.🚨 Outliers and Anomalies Detection")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, y="Yield")
ax.set_title("Yield Outliers")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, y="Production")
ax.set_title("Production Outliers")
st.pyplot(fig)

threshold_yield = df["Yield"].mean() + 3 * df["Yield"].std()
threshhold_production = df["Production"].mean() + 3 * df["Production"].std()
anomalies = df[(df["Yield"] > threshold_yield) | (df["Production"] > threshhold_production)]

st.subheader("📌 Anomaly Correlations")
st.write("Unusual Yield or Production values might be correlated with external factors such as policies or environmental changes.")
st.dataframe(anomalies)



# **4️⃣ Production Prediction**
st.subheader("📊 Production Prediction")
region = st.selectbox("Select Region", df['Area'].unique(), key="pred_region")
crop = st.selectbox("Select Crop", df['Item'].unique(), key="pred_crop")
year = st.number_input("Enter Year", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()) + 10, step=1)
area_harvested = st.number_input("Enter Area Harvested", min_value=0.0, step=0.1)

if st.button("Predict Production"):
    try:
        input_data = pd.DataFrame([[region, crop, year, area_harvested]], columns=['Area', 'Item', 'Year', 'Area_Harvested'])
        input_data = pd.get_dummies(input_data).reindex(columns=data_processor.scaler.feature_names_in_, fill_value=0)
        input_scaled = data_processor.scaler.transform(input_data)
        prediction = data_processor.model.predict(input_scaled)
        st.write(f"Predicted Production: {prediction[0]:,.2f} tons")
    except Exception as e:
        st.error(f"Error in prediction: {e}")