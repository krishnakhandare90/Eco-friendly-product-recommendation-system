import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="Eco-Friendly Product Recommender",
    page_icon="🌱",
    layout="wide"
)

# Loading the data 
@st.cache_data
def load_data():
    df = pd.read_csv( "D:\workshop19-7-25\eco-project .csv") 
    return df

@st.cache_data
def prepare_recommendation_system(df):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cert', OneHotEncoder(), ['Eco-Friendly Certification']),
            ('brand', OneHotEncoder(), ['Brand']),
            ('desc', TfidfVectorizer(stop_words='english'), 'Description')
        ],
        remainder='drop'
    )
    
    features = preprocessor.fit_transform(df)
    cosine_sim = cosine_similarity(features, features)
    
    indices = pd.Series(df.index, index=df['Product Name']).drop_duplicates()
    
    return cosine_sim, indices

#adding data for recommendation
df = load_data()
cosine_sim, indices = prepare_recommendation_system(df)

#function for recommendation
def get_recommendations(product_name, user_prefs=None, cosine_sim=cosine_sim):
    try:
        idx = indices[product_name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        if user_prefs:
            for i, score in sim_scores:
                product = df.iloc[i]
                if user_prefs.get('category') and product['Category'] == user_prefs['category']:
                    sim_scores[i] = (i, score * 1.3)
                if user_prefs.get('certification') and product['Eco-Friendly Certification'] == user_prefs['certification']:
                    sim_scores[i] = (i, score * 1.5)
                # addding price range 
                if user_prefs.get('max_price'):
                    if product['Price (USD)'] > float(user_prefs['max_price']):
                        sim_scores[i] = (i, 0)  
        
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  
        product_indices = [i[0] for i in sim_scores]
        
        return df.iloc[product_indices]
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

st.title("🌱 Eco-Friendly Product Recommender")
st.markdown("Discover sustainable alternatives based on your preferences!")

with st.sidebar:
    st.header("Your Preferences")
    category_pref = st.selectbox(
        "Preferred Category",
        ["All"] + sorted(df['Category'].unique().tolist())
    )
    cert_pref = st.selectbox(
        "Preferred Certification", 
        ["All"] + sorted(df['Eco-Friendly Certification'].unique().tolist())
    )
    max_price = st.slider(
        "Maximum Price (USD)",
        min_value=0,
        max_value=int(df['Price (USD)'].max()) + 10,
        value=int(df['Price (USD)'].max())
    )
    
    user_prefs = {
        'category': None if category_pref == "All" else category_pref,
        'certification': None if cert_pref == "All" else cert_pref,
        'max_price': max_price
    }

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Browse Products")
    selected_category = st.selectbox(
        "Filter by Category",
        ["All"] + sorted(df['Category'].unique().tolist())
    )
    
    if selected_category == "All":
        display_products = df
    else:
        display_products = df[df['Category'] == selected_category]
    
    selected_product = st.selectbox(
        "Select a Product",
        display_products['Product Name']
    )

with col2:
    if selected_product:
        st.subheader(f"Recommendations similar to: {selected_product}")
        
        product_details = df[df['Product Name'] == selected_product].iloc[0]
        
        with st.expander("View Selected Product Details", expanded=True):
            st.markdown(f"""
            **{product_details['Product Name']}**  
            *{product_details['Brand']}* | ${product_details['Price (USD)']}  
            **Category:** {product_details['Category']}  
            **Certification:** {product_details['Eco-Friendly Certification']}  
            **Description:** {product_details['Description']}
            """)
        
        recommendations = get_recommendations(selected_product, user_prefs)
        
        if not recommendations.empty:
            st.success("Here are some eco-friendly alternatives you might like:")
            for _, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="border-left: 4px solid #4CAF50; padding: 10px; margin: 10px 0; border-radius: 4px;">
                        <h4>{row['Product Name']}</h4>
                        <p><b>{row['Brand']}</b> | ${row['Price (USD)']} | {row['Eco-Friendly Certification']}</p>
                        <p>{row['Description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No recommendations found matching your criteria.")

# adding footer to web page 
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    color: #666;
    text-align: center;
}
</style>
<div class="footer">
    <p>🌍 Helping you make sustainable choices for a greener planet</p>
</div>
""", unsafe_allow_html=True)