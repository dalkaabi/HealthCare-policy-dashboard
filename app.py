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
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .policy-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .airline-highlight {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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

# Calculate key market statistics
@st.cache_data
def calculate_market_stats(df):
    """Calculate real market statistics from the data"""
    if df.empty:
        return {}
    
    stats = {}
    
    # Market share calculation
    airline_counts = df['airline'].value_counts()
    total_flights = len(df)
    market_share = (airline_counts / total_flights * 100).round(1)
    
    # Top 3 concentration
    top3_concentration = market_share.head(3).sum()
    
    # HHI calculation for routes
    route_hhi = []
    for route in df['route'].unique():
        route_data = df[df['route'] == route]
        airline_shares = route_data['airline'].value_counts(normalize=True)
        hhi = (airline_shares ** 2).sum() * 10000
        route_hhi.append({
            'route': route,
            'hhi': hhi,
            'flights': len(route_data),
            'airlines': len(airline_shares)
        })
    
    # Route concentration analysis
    high_concentration = len([r for r in route_hhi if r['hhi'] > 2500])
    total_routes = len(route_hhi)
    avg_hhi = np.mean([r['hhi'] for r in route_hhi])
    
    # Pricing statistics
    price_stats = {
        'min': df['price'].min(),
        'max': df['price'].max(),
        'mean': df['price'].mean(),
        'std': df['price'].std()
    }
    
    # Class analysis
    economy_pct = (df['class'] == 'Economy').mean() * 100
    business_pct = (df['class'] == 'Business').mean() * 100
    
    # Direct flights
    direct_flights_pct = (df['stops'] == 'zero').mean() * 100
    
    # Booking window pricing
    last_week = df[df['days_left'] <= 7]['price'].mean() if len(df[df['days_left'] <= 7]) > 0 else 0
    early_bird = df[df['days_left'] > 35]['price'].mean() if len(df[df['days_left'] > 35]) > 0 else 0
    price_premium = ((last_week - early_bird) / early_bird * 100) if early_bird > 0 else 0
    
    # Top routes by volume
    route_volumes = df['route'].value_counts().head(5)
    
    # Business class premium calculation
    economy_avg = df[df['class'] == 'Economy']['price'].mean()
    business_avg = df[df['class'] == 'Business']['price'].mean() if len(df[df['class'] == 'Business']) > 0 else 0
    business_premium = ((business_avg - economy_avg) / economy_avg * 100) if business_avg > 0 else 0
    
    stats = {
        'total_flights': total_flights,
        'total_routes': total_routes,
        'market_share': market_share.to_dict(),
        'top3_concentration': round(top3_concentration, 1),
        'high_concentration_routes': high_concentration,
        'high_concentration_pct': round(high_concentration / total_routes * 100, 1),
        'avg_hhi': round(avg_hhi, 0),
        'price_stats': price_stats,
        'economy_pct': round(economy_pct, 1),
        'business_pct': round(business_pct, 1),
        'direct_flights_pct': round(direct_flights_pct, 1),
        'price_premium': round(price_premium, 1),
        'top_routes': route_volumes.to_dict(),
        'business_premium': round(business_premium, 0),
        'route_hhi_data': route_hhi
    }
    
    return stats

# Load data
df = load_airline_data()

# Calculate market statistics
if not df.empty:
    market_stats = calculate_market_stats(df)
else:
    market_stats = {}

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
    default=sorted(df['route'].unique())[:10]  # Top 10 routes
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
if not filtered_df.empty and market_stats:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_price = filtered_df['price'].mean()
        baseline_price = market_stats['price_stats']['mean']
        st.metric("Average Fare", f"₹{avg_price:,.0f}", f"₹{avg_price-baseline_price:,.0f} vs overall")
    
    with col2:
        avg_duration = filtered_df['duration_hours'].mean()
        baseline_duration = df['duration_hours'].mean()
        st.metric("Average Duration", f"{avg_duration:.1f}h", f"{avg_duration-baseline_duration:.1f}h vs overall")
    
    with col3:
        total_flights = len(filtered_df)
        st.metric("Total Flights", f"{total_flights:,}", f"{total_flights/len(df)*100:.1f}% of total")
    
    with col4:
        direct_flights_pct = (filtered_df['stops'] == 'zero').mean() * 100
        baseline_direct = market_stats['direct_flights_pct']
        st.metric("Direct Flights", f"{direct_flights_pct:.1f}%", f"{direct_flights_pct-baseline_direct:.1f}% vs overall")
    
    with col5:
        avg_booking_window = filtered_df['days_left'].mean()
        baseline_booking = df['days_left'].mean()
        st.metric("Avg Booking Window", f"{avg_booking_window:.0f} days", f"{avg_booking_window-baseline_booking:.0f} vs overall")

# Display key market insights
if market_stats:
    st.markdown("### 📊 Market Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Market Leader:** {max(market_stats['market_share'], key=market_stats['market_share'].get)} 
        ({market_stats['market_share'][max(market_stats['market_share'], key=market_stats['market_share'].get)]}%)
        """)
    
    with col2:
        st.warning(f"""
        **High Concentration Routes:** {market_stats['high_concentration_routes']} of {market_stats['total_routes']} 
        ({market_stats['high_concentration_pct']}%)
        """)
    
    with col3:
        st.error(f"""
        **Avg HHI Index:** {market_stats['avg_hhi']:.0f} 
        (Above 2500 = High Concentration)
        """)

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
            filtered_df.sample(min(5000, len(filtered_df))),  # Sample for performance
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
    
    # Calculate Herfindahl-Hirschman Index (HHI) for each route
    route_hhi = []
    for route in filtered_df['route'].unique():
        route_data = filtered_df[filtered_df['route'] == route]
        airline_shares = route_data['airline'].value_counts(normalize=True)
        hhi = (airline_shares ** 2).sum() * 10000  # Multiply by 10000 for standard HHI scale
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

if not filtered_df.empty and market_stats:
    # Calculate key policy metrics using real data
    high_concentration_routes = market_stats['high_concentration_routes']
    total_routes = market_stats['total_routes']
    avg_hhi = market_stats['avg_hhi']
    top3_concentration = market_stats['top3_concentration']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="policy-box">
        <h4>🎯 Immediate Policy Actions</h4>
        <ul>
        <li><strong>Route Development:</strong> {high_concentration_routes} routes need competitive intervention (HHI > 2500)</li>
        <li><strong>Market Concentration:</strong> Top 3 airlines control {top3_concentration}% - monitor for anti-competitive practices</li>
        <li><strong>Slot Allocation:</strong> Redistribute slots on high-concentration routes to new entrants</li>
        <li><strong>Price Transparency:</strong> Mandate fare disclosure on routes with average HHI of {avg_hhi:.0f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="policy-box">
        <h4>📊 Data-Driven Insights</h4>
        <ul>
        <li><strong>Market Structure:</strong> {market_stats['high_concentration_pct']}% of routes highly concentrated</li>
        <li><strong>Pricing Efficiency:</strong> {market_stats['business_premium']}% business class premium indicates limited competition</li>
        <li><strong>Route Access:</strong> {market_stats['direct_flights_pct']}% direct flights suggest capacity constraints</li>
        <li><strong>Booking Patterns:</strong> {market_stats['price_premium']}% last-minute premium shows dynamic pricing issues</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Critical routes needing attention
    st.markdown("#### 🚨 Routes Requiring Regulatory Attention")
    
    # Get routes with highest HHI
    if 'route_hhi_data' in market_stats:
        high_hhi_routes = [r for r in market_stats['route_hhi_data'] if r['hhi'] > 3000]
        
        if high_hhi_routes:
            # Sort by HHI and take top 10
            high_hhi_routes = sorted(high_hhi_routes, key=lambda x: x['hhi'], reverse=True)[:10]
            
            critical_df = pd.DataFrame([{
                'Route': route['route'],
                'HHI Index': f"{route['hhi']:.0f}",
                'Airlines Serving': route['airlines'],
                'Total Flights': f"{route['flights']:,}",
                'Market Structure': 'Highly Concentrated' if route['hhi'] > 2500 else 'Moderately Concentrated'
            } for route in high_hhi_routes])
            
            st.dataframe(critical_df, use_container_width=True)
        else:
            st.success("✅ No routes show extremely high concentration levels (HHI > 3000)")
    
    # Market share analysis for policy
    st.markdown("#### ✈️ Airline-Specific Policy Considerations")
    
    market_share_data = market_stats['market_share']
    for airline, share in market_share_data.items():
        if share > 20:
            st.markdown(f"""
            <div class="airline-highlight">
            <strong>{airline}:</strong> Dominant market position with {share}% share - Requires competition oversight
            </div>
            """, unsafe_allow_html=True)
        elif share < 5:
            st.markdown(f"""
            <div class="airline-highlight">
            <strong>{airline}:</strong> Small market presence with {share}% share - Consider growth incentives
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
        if market_stats and 'route_hhi_data' in market_stats:
            hhi_csv = pd.DataFrame(market_stats['route_hhi_data']).to_csv(index=False)
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
