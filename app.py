import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Indian Aviation Policy Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         color: #2c3e50;
#         margin-bottom: 1rem;
#     }
#     .metric-container {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #1f77b4;
#     }
#     .policy-box {
#         background-color: #e8f4f8;
#         padding: 1.5rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #17a2b8;
#         margin: 1rem 0;
#     }
#     .airline-highlight {
#         background-color: #fff3cd;
#         padding: 0.5rem;
#         border-radius: 0.25rem;
#         border-left: 3px solid #ffc107;
#         margin: 0.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# Load airline data
@st.cache_data
def load_airline_data():
    try:
        # Read the airline data
        df = pd.read_csv('airlines_flights_data.csv')
        
        # Clean and process the data
        df = df.dropna()
        
        # Create route column
        df['route'] = df['source_city'] + ' → ' + df['destination_city']
        
        # Convert duration to proper format
        df['duration_hours'] = df['duration']
        
        # Create price categories
        df['price_category'] = pd.cut(df['price'], 
                                    bins=[0, 5000, 10000, 20000, float('inf')], 
                                    labels=['Budget', 'Mid-range', 'Premium', 'Luxury'])
        
        # Create booking window categories
        df['booking_window'] = pd.cut(df['days_left'], 
                                    bins=[0, 7, 21, 35, float('inf')], 
                                    labels=['Last Week', '1-3 Weeks', '3-5 Weeks', 'Early Bird'])
        
        # Map time periods to numerical values for analysis
        time_mapping = {
            'Early_Morning': 1, 'Morning': 2, 'Afternoon': 3, 
            'Evening': 4, 'Night': 5, 'Late_Night': 6
        }
        df['departure_numeric'] = df['departure_time'].map(time_mapping)
        df['arrival_numeric'] = df['arrival_time'].map(time_mapping)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_airline_data()

if df.empty:
    st.error("Unable to load airline data. Please check the file.")
    st.stop()

# Dashboard Title
st.markdown('<h1 class="main-header">✈️ Indian Aviation Policy Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Data-driven insights for aviation industry policy and regulation**")

# Sidebar filters
st.sidebar.header("📊 Dashboard Filters")

# Airline filter
selected_airlines = st.sidebar.multiselect(
    "Select Airlines:",
    options=sorted(df['airline'].unique()),
    default=sorted(df['airline'].unique())
)

# Route filter
selected_routes = st.sidebar.multiselect(
    "Select Routes:",
    options=sorted(df['route'].unique()),
    default=sorted(df['route'].unique())[:10]  
)

# Class filter
selected_classes = st.sidebar.multiselect(
    "Select Classes:",
    options=df['class'].unique(),
    default=df['class'].unique()
)

# Price range filter
price_range = st.sidebar.slider(
    "Price Range (₹):",
    min_value=int(df['price'].min()),
    max_value=int(df['price'].max()),
    value=(int(df['price'].min()), int(df['price'].max())),
    step=1000
)

# Filter data
filtered_df = df[
    (df['airline'].isin(selected_airlines)) & 
    (df['route'].isin(selected_routes)) &
    (df['class'].isin(selected_classes)) &
    (df['price'].between(price_range[0], price_range[1]))
]

# Key Metrics Row
if not filtered_df.empty:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_price = filtered_df['price'].mean()
        st.metric("Average Fare", f"₹{avg_price:,.0f}", f"₹{avg_price-df['price'].mean():,.0f} vs overall")
    
    with col2:
        avg_duration = filtered_df['duration_hours'].mean()
        st.metric("Average Duration", f"{avg_duration:.1f}h", f"{avg_duration-df['duration_hours'].mean():.1f}h vs overall")
    
    with col3:
        total_flights = len(filtered_df)
        st.metric("Total Flights", f"{total_flights:,}", f"{total_flights/len(df)*100:.1f}% of total")
    
    with col4:
        direct_flights_pct = (filtered_df['stops'] == 'zero').mean() * 100
        st.metric("Direct Flights", f"{direct_flights_pct:.1f}%", f"{direct_flights_pct-((df['stops'] == 'zero').mean() * 100):.1f}% vs overall")
    
    with col5:
        avg_booking_window = filtered_df['days_left'].mean()
        st.metric("Avg Booking Window", f"{avg_booking_window:.0f} days", f"{avg_booking_window-df['days_left'].mean():.0f} vs overall")

# Visualization Section
st.markdown("---")
st.markdown('<h2 class="sub-header">📈 Policy Analysis Visualizations</h2>', unsafe_allow_html=True)

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["🏢 Airline Analysis", "🛣️ Route Networks", "💰 Pricing Dynamics", "📊 Market Competition"])

with tab1:
    st.markdown("### Airline Performance and Market Share")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Market share analysis
        market_share = filtered_df['airline'].value_counts()
        fig_market = px.pie(
            values=market_share.values,
            names=market_share.index,
            title='Market Share by Airline (Flight Volume)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_market.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_market, use_container_width=True)
    
    with col2:
        # Airline metrics comparison
        airline_metrics = filtered_df.groupby('airline').agg({
            'price': 'mean',
            'duration_hours': 'mean',
            'days_left': 'mean'
        }).round(2)
        airline_metrics.columns = ['Avg Price (₹)', 'Avg Duration (h)', 'Avg Booking Window']
        
        st.markdown("**Airline Performance Metrics:**")
        st.dataframe(airline_metrics, use_container_width=True)
    
    # Airline comparison across multiple dimensions
    st.markdown("#### Multi-Dimensional Airline Comparison")
    
    # Prepare data for radar chart
    airline_comparison = filtered_df.groupby('airline').agg({
        'price': lambda x: 100 - (x.mean() - filtered_df['price'].min()) / (filtered_df['price'].max() - filtered_df['price'].min()) * 100,  # Lower price = higher score
        'duration_hours': lambda x: 100 - (x.mean() - filtered_df['duration_hours'].min()) / (filtered_df['duration_hours'].max() - filtered_df['duration_hours'].min()) * 100,  # Shorter duration = higher score
        'days_left': lambda x: (x.mean() - filtered_df['days_left'].min()) / (filtered_df['days_left'].max() - filtered_df['days_left'].min()) * 100  # More days = higher score (advance booking)
    }).round(2)
    
    # Add on-time performance proxy (stops as efficiency indicator)
    airline_comparison['efficiency'] = filtered_df.groupby('airline').apply(lambda x: (x['stops'] == 'zero').mean() * 100)
    
    # Create radar chart
    fig_radar = go.Figure()
    
    categories = ['Price Competitiveness', 'Time Efficiency', 'Advance Booking', 'Route Efficiency']
    
    for airline in airline_comparison.index:
        values = [
            airline_comparison.loc[airline, 'price'],
            airline_comparison.loc[airline, 'duration_hours'],
            airline_comparison.loc[airline, 'days_left'],
            airline_comparison.loc[airline, 'efficiency']
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=airline
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Airline Performance Comparison (0-100 scale)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.markdown("### Route Network Analysis")
    
    # Route performance analysis
    route_analysis = filtered_df.groupby('route').agg({
        'price': ['mean', 'min', 'max'],
        'duration_hours': 'mean',
        'airline': 'nunique',
        'flight': 'count'
    }).round(2)
    
    route_analysis.columns = ['Avg Price', 'Min Price', 'Max Price', 'Avg Duration', 'Airlines Serving', 'Daily Flights']
    route_analysis = route_analysis.sort_values('Daily Flights', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top routes by volume
        top_routes = route_analysis.head(10)
        fig_routes = px.bar(
            x=top_routes['Daily Flights'],
            y=top_routes.index,
            orientation='h',
            title='Top 10 Routes by Flight Volume',
            labels={'x': 'Number of Flights', 'y': 'Route'},
            color=top_routes['Avg Price'],
            color_continuous_scale='viridis'
        )
        fig_routes.update_layout(height=500)
        st.plotly_chart(fig_routes, use_container_width=True)
    
    with col2:
        # Route competition analysis
        fig_competition = px.scatter(
            route_analysis.reset_index(),
            x='Airlines Serving',
            y='Avg Price',
            size='Daily Flights',
            hover_data=['route', 'Avg Duration'],
            title='Route Competition vs Price',
            labels={'Airlines Serving': 'Number of Airlines', 'Avg Price': 'Average Price (₹)'}
        )
        st.plotly_chart(fig_competition, use_container_width=True)
    
    # Route efficiency analysis
    st.markdown("#### Route Efficiency Metrics")
    
    # Calculate route efficiency (flights per hour of duration)
    route_analysis['Efficiency Score'] = route_analysis['Daily Flights'] / route_analysis['Avg Duration']
    route_analysis['Price per Hour'] = route_analysis['Avg Price'] / route_analysis['Avg Duration']
    
    efficiency_df = route_analysis[['Avg Duration', 'Efficiency Score', 'Price per Hour']].head(15)
    
    fig_efficiency = px.bar(
        efficiency_df.reset_index(),
        x='route',
        y=['Avg Duration', 'Efficiency Score', 'Price per Hour'],
        title='Route Efficiency Analysis (Top 15 Routes)',
        barmode='group'
    )
    fig_efficiency.update_xaxes(tickangle=45)
    st.plotly_chart(fig_efficiency, use_container_width=True)

with tab3:
    st.markdown("### Pricing Strategy Analysis")
    
    # Price distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by class
        fig_price_class = px.box(
            filtered_df,
            x='class',
            y='price',
            color='class',
            title='Price Distribution by Class',
            labels={'price': 'Price (₹)', 'class': 'Travel Class'}
        )
        st.plotly_chart(fig_price_class, use_container_width=True)
    
    with col2:
        # Price vs booking window
        fig_price_booking = px.scatter(
            filtered_df.sample(min(5000, len(filtered_df))),
            x='days_left',
            y='price',
            color='airline',
            title='Price vs Booking Window',
            labels={'days_left': 'Days Before Departure', 'price': 'Price (₹)'},
            trendline="lowess"
        )
        st.plotly_chart(fig_price_booking, use_container_width=True)
    
    # Advanced pricing analysis
    st.markdown("#### Dynamic Pricing Insights")
    
    # Price by departure time
    time_price = filtered_df.groupby(['departure_time', 'airline'])['price'].mean().reset_index()
    fig_time_price = px.bar(
        time_price,
        x='departure_time',
        y='price',
        color='airline',
        title='Average Price by Departure Time',
        barmode='group'
    )
    st.plotly_chart(fig_time_price, use_container_width=True)
    
    # Stops impact on pricing
    stops_price = filtered_df.groupby(['stops', 'class']).agg({
        'price': 'mean',
        'duration_hours': 'mean'
    }).reset_index()
    
    fig_stops = px.bar(
        stops_price,
        x='stops',
        y='price',
        color='class',
        title='Price Impact of Connections',
        barmode='group',
        labels={'stops': 'Number of Stops', 'price': 'Average Price (₹)'}
    )
    st.plotly_chart(fig_stops, use_container_width=True)

with tab4:
    st.markdown("### Market Competition Analysis")
    
    # Competition intensity by route
    competition_data = filtered_df.groupby('route').agg({
        'airline': 'nunique',
        'price': ['mean', 'std'],
        'flight': 'count'
    }).round(2)
    
    competition_data.columns = ['Airlines_Count', 'Avg_Price', 'Price_Std', 'Flight_Count']
    competition_data['Competition_Level'] = pd.cut(
        competition_data['Airlines_Count'],
        bins=[0, 2, 4, 6, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Competition levels distribution
        comp_dist = competition_data['Competition_Level'].value_counts()
        fig_comp_dist = px.pie(
            values=comp_dist.values,
            names=comp_dist.index,
            title='Distribution of Competition Levels',
            color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        )
        st.plotly_chart(fig_comp_dist, use_container_width=True)
    
    with col2:
        # Price variance vs competition
        fig_price_var = px.scatter(
            competition_data.reset_index(),
            x='Airlines_Count',
            y='Price_Std',
            size='Flight_Count',
            hover_data=['route'],
            title='Price Variance vs Competition Level',
            labels={'Airlines_Count': 'Number of Airlines', 'Price_Std': 'Price Standard Deviation'}
        )
        st.plotly_chart(fig_price_var, use_container_width=True)
    
    # Market concentration analysis
    st.markdown("#### Market Concentration by Route")
    
    route_hhi = []
    for route in filtered_df['route'].unique():
        route_data = filtered_df[filtered_df['route'] == route]
        airline_shares = route_data['airline'].value_counts(normalize=True)
        hhi = (airline_shares ** 2).sum() * 10000 
        route_hhi.append({'route': route, 'HHI': hhi, 'flights': len(route_data)})
    
    hhi_df = pd.DataFrame(route_hhi)
    hhi_df['Market_Structure'] = pd.cut(
        hhi_df['HHI'],
        bins=[0, 1500, 2500, 10000],
        labels=['Competitive', 'Moderately Concentrated', 'Highly Concentrated']
    )
    
    fig_hhi = px.scatter(
        hhi_df,
        x='flights',
        y='HHI',
        color='Market_Structure',
        hover_data=['route'],
        title='Market Concentration Analysis (HHI Index)',
        labels={'flights': 'Number of Flights', 'HHI': 'Herfindahl-Hirschman Index'}
    )
    fig_hhi.add_hline(y=1500, line_dash="dash", annotation_text="Competitive Threshold")
    fig_hhi.add_hline(y=2500, line_dash="dash", annotation_text="Concentration Threshold")
    st.plotly_chart(fig_hhi, use_container_width=True)

# Policy Recommendations Section
st.markdown("---")
st.markdown('<h2 class="sub-header">📋 Policy Recommendations</h2>', unsafe_allow_html=True)

if not filtered_df.empty:
    # Calculate key policy metrics
    avg_hhi = pd.DataFrame(route_hhi)['HHI'].mean()
    high_price_routes = len(route_analysis[route_analysis['Avg Price'] > route_analysis['Avg Price'].quantile(0.8)])
    monopoly_routes = len([r for r in route_hhi if r['HHI'] > 2500])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="policy-box">
        <h4>🎯 Immediate Policy Actions</h4>
        <ul>
        <li><strong>Route Development:</strong> Incentivize new entrants on high-concentration routes</li>
        <li><strong>Pricing Regulation:</strong> Monitor routes with HHI > 2500 for price manipulation</li>
        <li><strong>Slot Allocation:</strong> Ensure fair distribution of airport slots among carriers</li>
        <li><strong>Consumer Protection:</strong> Implement transparent pricing and booking policies</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="policy-box">
        <h4>📊 Data-Driven Insights</h4>
        <ul>
        <li><strong>Market Concentration:</strong> Average HHI of {:.0f} indicates moderate concentration</li>
        <li><strong>Route Competition:</strong> {} routes show high concentration levels</li>
        <li><strong>Price Efficiency:</strong> Significant price variance across similar routes</li>
        <li><strong>Booking Patterns:</strong> Early booking incentives vary significantly by airline</li>
        </ul>
        </div>
        """.format(avg_hhi, monopoly_routes), unsafe_allow_html=True)
    
    # Critical routes needing attention
    st.markdown("#### 🚨 Routes Requiring Regulatory Attention")
    
    critical_routes = []
    for route_info in route_hhi:
        if route_info['HHI'] > 2500: 
            route_data = route_analysis.loc[route_info['route']]
            critical_routes.append({
                'Route': route_info['route'],
                'HHI': route_info['HHI'],
                'Airlines': int(route_data['Airlines Serving']),
                'Avg Price': f"₹{route_data['Avg Price']:,.0f}",
                'Market Structure': 'Highly Concentrated'
            })
    
    if critical_routes:
        critical_df = pd.DataFrame(critical_routes)
        st.dataframe(critical_df, use_container_width=True)
    else:
        st.success("✅ No routes show concerning levels of market concentration")
    
    # Airline-specific recommendations
    st.markdown("#### ✈️ Airline-Specific Policy Considerations")
    
    airline_summary = filtered_df.groupby('airline').agg({
        'route': 'nunique',
        'price': 'mean',
        'flight': 'count'
    }).round(0)
    airline_summary.columns = ['Routes Served', 'Avg Price', 'Total Flights']
    airline_summary['Market Share %'] = (airline_summary['Total Flights'] / airline_summary['Total Flights'].sum() * 100).round(1)
    
    for airline in airline_summary.index:
        market_share = airline_summary.loc[airline, 'Market Share %']
        if market_share > 25:
            st.markdown(f"""
            <div class="airline-highlight">
            <strong>{airline}:</strong> Market leader with {market_share}% share - Monitor for anti-competitive practices
            </div>
            """, unsafe_allow_html=True)
        elif market_share < 5:
            st.markdown(f"""
            <div class="airline-highlight">
            <strong>{airline}:</strong> Small player with {market_share}% share - Consider growth incentives
            </div>
            """, unsafe_allow_html=True)

# Download section
st.markdown("---")
st.markdown("### 📥 Export Data and Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Download Filtered Dataset"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"airline_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Download Route Analysis"):
        route_csv = route_analysis.to_csv()
        st.download_button(
            label="Download Route Data",
            data=route_csv,
            file_name=f"route_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("Download Competition Metrics"):
        hhi_csv = pd.DataFrame(route_hhi).to_csv(index=False)
        st.download_button(
            label="Download HHI Data",
            data=hhi_csv,
            file_name=f"market_concentration_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
<p>Indian Aviation Policy Dashboard | Created for LIS 407 Final Project<br>
Data Source: <a href="https://www.kaggle.com/datasets/rohitgrewal/airlines-flights-data">Kaggle Airlines Flights Dataset</a> | 300K+ flight records</p>
</div>
""", unsafe_allow_html=True)